"""
ValidadorExcelNovoFormato
=========================
Processa um SEGUNDO layout de Excel, reutilizando integralmente o pipeline
de YOLO + OCR + validaÃ§Ã£o de placas jÃ¡ existente em `app_excel.py`.

DiferenÃ§as em relaÃ§Ã£o ao formato original
------------------------------------------
  Formato original  â†’ mÃºltiplos blocos por arquivo, bloco inicia quando
                       col_A Ã© inteiro positivo, 17+ colunas de metadados.

  Novo formato      â†’ UMA passagem por arquivo.
                       Dados fixos na LINHA 6 (Ã­ndice 0-based = 5).
                       Imagens ancoradas a partir da linha 8.
                       Mapeamento de colunas diferente (col_B = ID, etc.).

O que este arquivo faz
-----------------------
  â€¢ Herda ValidadorExcelPro sem modificar nenhuma linha do original.
  â€¢ Sobrescreve APENAS dois mÃ©todos:
      _ler_blocos()   â†’ lÃª a Ãºnica passagem do novo layout.
      processar()     â†’ adapta o pipeline e o relatÃ³rio final ao novo formato.
  â€¢ Toda a lÃ³gica de OCR, YOLO, prÃ©-processamento, Levenshtein e
    mapeamento XML permanece 100 % intacta (herdada diretamente).

Estrutura do novo Excel (sheet "Fotos")
----------------------------------------
  L001-L004  â†’ cabeÃ§alhos / tÃ­tulo   (ignorados)
  L005       â†’ rÃ³tulos das colunas   (ignorados)
  L006       â†’ dados da passagem:
                 col_B  = ID do sistema de arrecadaÃ§Ã£o
                 col_C  = DATA   (datetime)
                 col_D  = HORA   (time)
                 col_E  = SP
                 col_F  = KM+M
                 col_G  = Nome da praÃ§a
                 col_H  = NÂº da pista
                 col_I  = Sentido
                 col_J  = MunicÃ­pio
                 col_K  = CÃ³digo
                 col_L  = PLACA
                 col_M  = Marca/Modelo
                 col_N  = Quantidade de eixos
                 col_O  = Valor da tarifa
                 col_P  = ResponsÃ¡vel
  L007       â†’ aviso textual         (ignorado)
  L008+      â†’ imagens ancoradas via drawing XML
"""

import os
from glob import glob

import pandas as pd
from openpyxl import load_workbook
from tqdm import tqdm
from config import DOWNLOADS_DIR, MODEL_PATH, RESULTS_EVASAO_DIR, ensure_parent

# â”€â”€ Importa a classe base sem alterar uma vÃ­rgula dela â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from app_excel_spvias import ValidadorExcelPro


# ExtensÃµes de imagem que o pipeline consegue processar.
# O formato original ignora apenas o logo (.png fixo); aqui excluÃ­mos tambÃ©m
# arquivos .emf (metafile do Windows), que nÃ£o sÃ£o fotos de veÃ­culo.
_EXTENSOES_IGNORADAS = {'.emf', '.wmf'}


class ValidadorNovoFormato(ValidadorExcelPro):
    """
    Subclasse especializada no segundo layout de Excel.

    Herda e reutiliza sem alteraÃ§Ã£o:
      _processar_imagem_yolo_ocr  Â· _avaliar_bloco  Â· _mapear_imagens_xml
      _normalizar_placa_ocr       Â· _distancia_placas Â· _pre_processar_imagem
      _limpar_texto_ocr           Â· _corrigir_por_mascara Â· _descobrir_mascara
    """

    # â”€â”€ Leitura do novo layout â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _ler_blocos(self, ws) -> list[dict]:
        """
        LÃª a ÃšNICA passagem do novo formato.

        A passagem estÃ¡ sempre na linha 6 (row_idx = 6 na contagem 1-based).
        Retorna uma lista com um Ãºnico dicionÃ¡rio, no mesmo schema esperado
        pelo restante do pipeline (_avaliar_bloco, processar).

        O campo 'linha' indica onde comeÃ§am as imagens (linha 7+), e
        'linha_fim' Ã© definido como 99999 â€” hÃ¡ apenas um bloco por arquivo.
        """
        LINHA_DADOS = 6  # linha fixa do novo formato

        bloco = None
        for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
            if row_idx != LINHA_DADOS:
                continue

            # Mapeamento de colunas do novo layout (0-based)
            # col_A (0) = None (cÃ©lula vazia)
            # col_B (1) = ID do sistema de arrecadaÃ§Ã£o
            # col_C (2) = DATA
            # col_D (3) = HORA
            # col_E (4) = SP
            # col_F (5) = KM+M
            # col_G (6) = Nome da praÃ§a
            # col_H (7) = NÂº da pista
            # col_I (8) = Sentido
            # col_J (9) = MunicÃ­pio
            # col_K (10) = CÃ³digo
            # col_L (11) = PLACA  â† campo crÃ­tico para OCR
            # col_M (12) = Marca/Modelo
            # col_N (13) = Quantidade de eixos
            # col_O (14) = Valor da tarifa
            # col_P (15) = ResponsÃ¡vel
            bloco = {
                'seq':         1,
                'linha':       LINHA_DADOS,
                'linha_fim':   99999,
                'id':          str(row[1])  if row[1]  is not None else '',
                'data':        row[2],
                'hora':        row[3],
                'sp':          str(row[4])  if row[4]  is not None else '',
                'km':          str(row[5])  if row[5]  is not None else '',
                'praca':       str(row[6])  if row[6]  is not None else '',
                'pista':       row[7],
                'sentido':     str(row[8])  if row[8]  is not None else '',
                'municipio':   str(row[9])  if row[9]  is not None else '',
                'codigo':      row[10],
                'placa':       str(row[11]) if row[11] is not None else '',
                'modelo':      str(row[12]) if row[12] is not None else '',
                'eixos':       row[13],
                'tarifa':      row[14],
                'responsavel': str(row[15]) if row[15] is not None else '',
            }
            break  # uma Ãºnica linha de dados â€” encerra imediatamente

        return [bloco] if bloco else []

    # â”€â”€ Mapeamento de imagens adaptado para ignorar .emf / .wmf â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _mapear_imagens_xml(self, caminho_excel: str) -> dict[int, list[bytes]]:
        """
        ExtensÃ£o do mÃ©todo da classe base: filtra extensÃµes nÃ£o processÃ¡veis
        (.emf, .wmf) que aparecem neste formato de Excel como logotipos/marcas.

        Toda a lÃ³gica de leitura do ZIP e parsing dos XMLs Ã© herdada;
        reescrevemos apenas o filtro final de extensÃ£o.
        """
        import zipfile
        import xml.etree.ElementTree as ET

        # Namespaces (copiados do mÃ³dulo pai para autonomia do mÃ©todo)
        _NS = {
            'xdr': 'http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing',
            'a':   'http://schemas.openxmlformats.org/drawingml/2006/main',
            'r':   'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
            'rel': 'http://schemas.openxmlformats.org/package/2006/relationships',
        }

        mapa: dict[int, list[bytes]] = {}

        with zipfile.ZipFile(caminho_excel, 'r') as z:
            arquivos = z.namelist()

            rels_sheet = 'xl/worksheets/_rels/sheet1.xml.rels'
            if rels_sheet not in arquivos:
                print('AVISO: sheet1.xml.rels nÃ£o encontrado.')
                return mapa

            sheet_rels = ET.fromstring(z.read(rels_sheet).decode())
            drawing_path = None
            for rel in sheet_rels.findall('rel:Relationship', _NS):
                if 'drawing' in rel.get('Type', '').lower():
                    target = rel.get('Target').lstrip('./')
                    drawing_path = 'xl/' + target.lstrip('/')
                    break

            if not drawing_path or drawing_path not in arquivos:
                print(f'AVISO: drawing nÃ£o encontrado ({drawing_path}).')
                return mapa

            partes            = drawing_path.rsplit('/', 1)
            drawing_rels_path = partes[0] + '/_rels/' + partes[1] + '.rels'
            if drawing_rels_path not in arquivos:
                print(f'AVISO: {drawing_rels_path} nÃ£o encontrado.')
                return mapa

            rels_drawing = ET.fromstring(z.read(drawing_rels_path).decode())
            rid_para_arquivo = {
                r.get('Id'): r.get('Target', '').split('/')[-1]
                for r in rels_drawing.findall('rel:Relationship', _NS)
            }

            bytes_midia = {
                arq.split('/')[-1]: z.read(arq)
                for arq in arquivos
                if arq.startswith('xl/media/')
            }

            drawing_root = ET.fromstring(z.read(drawing_path).decode())
            for anchor in drawing_root.findall('xdr:twoCellAnchor', _NS):
                from_elem = anchor.find('xdr:from', _NS)
                if from_elem is None:
                    continue
                row_inicio = int(from_elem.find('xdr:row', _NS).text) + 1

                pic = anchor.find('xdr:pic', _NS)
                if pic is None:
                    continue
                blip = pic.find('.//a:blip', _NS)
                if blip is None:
                    continue

                rid = blip.get(
                    '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed'
                )
                nome_arquivo = rid_para_arquivo.get(rid, '')
                if not nome_arquivo:
                    continue

                # â”€â”€ Filtro de extensÃ£o â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                extensao = os.path.splitext(nome_arquivo)[1].lower()
                if extensao in _EXTENSOES_IGNORADAS:
                    continue

                img_bytes = bytes_midia.get(nome_arquivo)
                if img_bytes:
                    mapa.setdefault(row_inicio, []).append(img_bytes)

        return mapa

    # â”€â”€ Pipeline principal adaptado â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def processar(self, pasta_excel: str, caminho_saida: str):
        """
        Pipeline completo para o novo formato.

        DiferenÃ§as em relaÃ§Ã£o ao mÃ©todo pai:
          â€¢ Usa sheet 'Fotos' (primeiro sheet) ao invÃ©s de ws.active genÃ©rico
            (neste formato o active jÃ¡ Ã© o correto, mas explicitamos por clareza).
          â€¢ Coleta as imagens usando linha_fim=99999 para cobrir todo o arquivo.
          â€¢ RelatÃ³rio final inclui os mesmos campos de OCR do formato original,
            acrescido dos metadados do novo layout (ID, HORA, SP, KM, etc.).
        """
        arquivos_excel = glob(os.path.join(pasta_excel, '*.xlsx'))
        if not arquivos_excel:
            print('Nenhum arquivo .xlsx encontrado na pasta informada.')
            return

        registros = []

        for caminho_excel in arquivos_excel:
            print(f'\nProcessando: {caminho_excel}')

            print('Carregando planilha (sheet "Fotos")...')
            wb = load_workbook(caminho_excel, data_only=True)

            # Garante que lemos a sheet correta mesmo que o active mude
            ws = wb['Fotos'] if 'Fotos' in wb.sheetnames else wb.active

            print('Lendo passagem Ãºnica do novo formato...')
            blocos = self._ler_blocos(ws)

            if not blocos:
                print(f'  AVISO: nenhuma passagem encontrada em {caminho_excel}. Pulando.')
                continue

            print('Mapeando imagens via XML interno...')
            mapa_bytes = self._mapear_imagens_xml(caminho_excel)

            # Associa TODAS as imagens do arquivo ao Ãºnico bloco existente
            for b in blocos:
                imgs = []
                for linha_ancora, lista_bytes in mapa_bytes.items():
                    # linha_ancora >= linha_dados cobre todas as fotos do arquivo
                    if linha_ancora >= b['linha']:
                        imgs.extend(lista_bytes)
                b['_imagens_bytes'] = imgs

            print(f'  1 passagem | {sum(len(b["_imagens_bytes"]) for b in blocos)} imagens encontradas.')

            print('Iniciando anÃ¡lise YOLO + OCR...')

            for b in tqdm(blocos):
                # _avaliar_bloco Ã© herdado integralmente â€” nenhuma alteraÃ§Ã£o
                resultado = self._avaliar_bloco(
                    b['placa'],
                    b['_imagens_bytes']
                )

                registros.append({
                    # â”€â”€ IdentificaÃ§Ã£o do arquivo â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                    'Arquivo':        os.path.basename(caminho_excel),

                    # â”€â”€ Dados da passagem (novo layout) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                    'ID':             b['id'],
                    'Data':           b['data'],
                    'Hora':           b['hora'],
                    'PraÃ§a':          b['praca'],
                    'Sentido':        b['sentido'],
                    'MunicÃ­pio':      b['municipio'],
                    'CÃ³digo':         b['codigo'],
                    'Placa Esperada': b['placa'],
                    'Eixos':          b['eixos'],

                    # â”€â”€ Colunas de OCR (idÃªnticas ao formato original) â”€â”€â”€â”€
                    'Placa OCR':      resultado['placa_ocr'],
                    'OCR Bruto':      resultado['ocr_bruto'],
                    'Status OCR':     resultado['status'],
                    'DiferenÃ§a OCR':  resultado['diferenca'],
                    'Imagem Usada':   resultado['imagem_usada'],
                    'Total Imagens':  resultado['total_imgs'],
                    'Imagens c/ OCR': resultado['imgs_com_ocr'],
                })

        if not registros:
            print('\nNenhum registro gerado. Verifique os arquivos de entrada.')
            return

        df = pd.DataFrame(registros)
        df.to_excel(caminho_saida, index=False)

        # â”€â”€ Resumo â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        total     = len(df)
        aprovados = (df['Status OCR'] == 'Aprovado - busca nas imagens').sum()
        revisar   = (df['Status OCR'] == 'Revisar').sum()
        sem_img   = (df['Status OCR'] == 'Revisar - sem imagem').sum()

        print(f"\n{'='*54}")
        print(f"  Resultado final: {total} passagens")
        print(f"  Aprovado:        {aprovados:3d} ({aprovados/total*100:.0f}%)")
        print(f"  Revisar:         {revisar:3d} ({revisar/total*100:.0f}%)")
        print(f"  Sem imagem:      {sem_img:3d} ({sem_img/total*100:.0f}%)")
        print(f"{'='*54}")
        print(f"  Arquivo gerado: {caminho_saida}")


# â”€â”€ Ponto de entrada â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == '__main__':

    PATH_YOLO = MODEL_PATH

    PASTA_EXCELS = DOWNLOADS_DIR / 'evasoes_colinas'

    PATH_OUT = ensure_parent(RESULTS_EVASAO_DIR / 'resultado_colinas.xlsx')

    validador = ValidadorNovoFormato(
        yolo_weights_path=PATH_YOLO,
        usar_gpu=True,
    )

    validador.processar(
        PASTA_EXCELS,
        PATH_OUT,
    )



