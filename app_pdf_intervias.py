"""
ValidadorPDFNovoFormato
=======================
Processa o layout de PDF da Intervias / Arteris, reutilizando integralmente
o pipeline de YOLO + OCR + validaÃ§Ã£o de placas jÃ¡ existente em `app_excel.py`.

DiferenÃ§a em relaÃ§Ã£o aos processadores anteriores
--------------------------------------------------
  app_excel.py              â†’ lÃª mÃºltiplos blocos de um .xlsx
  app_excel_novo_formato.py â†’ lÃª um Ãºnico bloco de um .xlsx
  app_pdf_novo_formato.py   â†’ lÃª um Ãºnico bloco de um .pdf  â† ESTE ARQUIVO

Estrutura do PDF
-----------------
  O PDF contÃ©m uma tabela com os dados da passagem e imagens embutidas.

  Campos extraÃ­dos da tabela (linha de dados):
    col[0]  = ID do sistema de arrecadaÃ§Ã£o
    col[1]  = DATA  (string "DD/MM/YYYY")
    col[2]  = HORA  (string "HH:MM:SS")
    col[3]  = SP
    col[4]  = KM+M
    col[5]  = Nome da praÃ§a de pedÃ¡gio
    col[6]  = NÂº da pista
    col[7]  = Sentido
    col[8]  = MunicÃ­pio
    col[9]  = CÃ³digo
    col[10] = PLACA  â† campo crÃ­tico para OCR
    col[11] = Marca/Modelo
    col[12] = Quantidade de eixos
    col[13] = Valor da tarifa
    col[14] = ResponsÃ¡vel (NOME)

  Imagens:
    O PDF embute as fotos do veÃ­culo diretamente na pÃ¡gina.
    SÃ£o identificadas por tamanho: imagens >= MIN_AREA_FOTO pxÂ² sÃ£o fotos;
    imagens menores sÃ£o logos/marcas d'Ã¡gua e ignoradas.

O que este arquivo faz
-----------------------
  â€¢ Herda ValidadorExcelPro sem modificar nenhuma linha do original.
  â€¢ Implementa APENAS a leitura do PDF (dados + imagens).
  â€¢ Toda a lÃ³gica de YOLO, OCR, prÃ©-processamento e Levenshtein Ã© herdada.

DependÃªncias adicionais
------------------------
  pip install pymupdf pdfplumber
"""

import os
import re
from glob import glob
from io import BytesIO
from datetime import datetime

import fitz          # PyMuPDF â€” extraÃ§Ã£o de imagens embutidas
import pdfplumber    # extraÃ§Ã£o de texto/tabela com layout
import pandas as pd
from tqdm import tqdm

# â”€â”€ Importa a classe base sem alterar uma vÃ­rgula dela â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

from app_excel_spvias import ValidadorExcelPro
import os
from pathlib import Path
from config import DOWNLOADS_DIR, MODEL_PATH, RESULTS_EVASAO_DIR, ensure_parent


# Ãrea mÃ­nima (largura Ã— altura em pixels) para considerar uma imagem como
# foto do veÃ­culo. Logos e marcas d'Ã¡gua tÃªm Ã¡rea muito menor.
_MIN_AREA_FOTO_PX = 100_000   # ex: 1280Ã—720 â‰ˆ 921 600 >> 100 000


class ValidadorPDFNovoFormato(ValidadorExcelPro):
    """
    Subclasse especializada no formato PDF da Intervias/Arteris.

    Herda e reutiliza sem alteraÃ§Ã£o:
      _processar_imagem_yolo_ocr  Â· _avaliar_bloco
      _normalizar_placa_ocr       Â· _distancia_placas Â· _pre_processar_imagem
      _limpar_texto_ocr           Â· _corrigir_por_mascara Â· _descobrir_mascara
      _mapear_imagens_xml         (nÃ£o usado aqui â€” substituÃ­do por _extrair_imagens_pdf)
    """

    # â”€â”€ Leitura dos dados da passagem â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _ler_passagem_pdf(self, caminho_pdf: str) -> dict | None:

        with pdfplumber.open(caminho_pdf) as pdf:
            texto = pdf.pages[0].extract_text()

        if not texto:
            print(f'AVISO: texto nÃ£o encontrado em {caminho_pdf}')
            return None

        linhas = [
            l.strip()
            for l in texto.split('\n')
            if l.strip()
        ]

        linha_dados = None

        # procura a linha que comeÃ§a com o ID gigante
        for linha in linhas:

            if re.match(r'^([A-Z0-9]{20,})', linha):
                linha_dados = linha
                break

        if not linha_dados:
            print(f'AVISO: linha da passagem nÃ£o encontrada.')
            return None

        """
        Exemplo real:

        080102SAU202601011739313750
        01/01/2026
        14:39:31
        330
        KM 215+000
        PIRASSUNUNGA
        2
        SUL
        PIRASSUNUNGA
        68870
        CWV9G20
        TOYOTA /HILUX CDSRXA4FD
        1
        11,20
        R$
        JONATHAN NEVES
        """

        tokens = linha_dados.split()

        try:

            id_sistema = tokens[0]
            data = tokens[1]
            hora = tokens[2]
            sp = tokens[3]

            # KM pode vir separado
            km = f'{tokens[4]} {tokens[5]}'

            praca = tokens[6]

            pista = tokens[7]

            sentido = tokens[8]

            municipio = tokens[9]

            codigo = tokens[10]

            placa = tokens[11]

            eixos = tokens[-4]

            tarifa = tokens[-3]

            responsavel = ' '.join(tokens[-1:])

            modelo_tokens = tokens[12:-4]
            modelo = ' '.join(modelo_tokens)

        except Exception as e:

            print(f'Erro ao interpretar linha:')
            print(linha_dados)
            print(e)

            return None

        return {

            'id': id_sistema,
            'data': data,
            'hora': hora,
            'sp': sp,
            'km': km,
            'praca': praca,
            'pista': pista,
            'sentido': sentido,
            'municipio': municipio,
            'codigo': codigo,
            'placa': placa,
            'modelo': modelo,
            'eixos': eixos,
            'tarifa': tarifa,
            'responsavel': responsavel,
        }

    # â”€â”€ ExtraÃ§Ã£o de imagens embutidas no PDF â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _extrair_imagens_pdf(self, caminho_pdf: str) -> list[bytes]:
        """
        Extrai as fotos do veÃ­culo embutidas no PDF usando PyMuPDF.

        CritÃ©rio de filtragem: apenas imagens cuja Ã¡rea (largura Ã— altura)
        em pixels supera _MIN_AREA_FOTO_PX sÃ£o consideradas fotos do veÃ­culo.
        Logos, marcas d'Ã¡gua e Ã­cones tÃªm Ã¡rea muito menor e sÃ£o descartados.

        Retorna uma lista de bytes prontos para _avaliar_bloco.
        """
        fotos: list[bytes] = []

        doc = fitz.open(caminho_pdf)
        page = doc[0]   # sempre uma Ãºnica pÃ¡gina neste formato

        for img_info in page.get_images(full=True):
            xref = img_info[0]
            img_data = doc.extract_image(xref)

            largura = img_data['width']
            altura  = img_data['height']
            area    = largura * altura

            if area < _MIN_AREA_FOTO_PX:
                continue  # descarta logos e marcas d'Ã¡gua

            # Converte para JPEG em memÃ³ria se jÃ¡ nÃ£o for JPEG/PNG
            # (_processar_imagem_yolo_ocr aceita qualquer formato via PIL)
            fotos.append(img_data['image'])

        doc.close()
        return fotos

    # â”€â”€ Pipeline principal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def processar(self, pasta_pdf: str, caminho_saida: str):
        """
        Pipeline completo para o formato PDF.

          1. LÃª todos os .pdf da pasta.
          2. Para cada PDF: extrai dados da tabela + fotos embutidas.
          3. Chama _avaliar_bloco (herdado) com a placa e as fotos.
          4. Consolida todos os registros e salva o Excel final.
        """
        arquivos_pdf = glob(os.path.join(pasta_pdf, '*.pdf'))
        if not arquivos_pdf:
            print('Nenhum arquivo .pdf encontrado na pasta informada.')
            return

        registros = []

        for caminho_pdf in arquivos_pdf:
            print(f'\nProcessando: {caminho_pdf}')

            print('Extraindo dados da tabela...')
            passagem = self._ler_passagem_pdf(caminho_pdf)
            if passagem is None:
                print('  Pulando â€” dados nÃ£o encontrados.')
                continue

            print('Extraindo fotos embutidas...')
            fotos = self._extrair_imagens_pdf(caminho_pdf)
            print(f'  {len(fotos)} foto(s) encontrada(s).')

            print('Iniciando anÃ¡lise YOLO + OCR...')
            # _avaliar_bloco Ã© herdado integralmente â€” nenhuma alteraÃ§Ã£o
            resultado = self._avaliar_bloco(passagem['placa'], fotos)

            registros.append({
                # â”€â”€ IdentificaÃ§Ã£o â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                'Arquivo':        os.path.basename(caminho_pdf),

                # â”€â”€ Dados da passagem â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                'ID':             passagem['id'],
                'Data':           passagem['data'],
                'Hora':           passagem['hora'],
                'SP':             passagem['sp'],
                'KM':             passagem['km'],
                'PraÃ§a':          passagem['praca'],
                'Pista':          passagem['pista'],
                'Sentido':        passagem['sentido'],
                'MunicÃ­pio':      passagem['municipio'],
                'CÃ³digo':         passagem['codigo'],
                'Placa Esperada': passagem['placa'],
                'Modelo':         passagem['modelo'],
                'Eixos':          passagem['eixos'],
                'Tarifa':         passagem['tarifa'],
                'ResponsÃ¡vel':    passagem['responsavel'],

                # â”€â”€ Colunas de OCR (idÃªnticas aos formatos anteriores) â”€â”€â”€â”€â”€
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

    PASTA_PDFS = DOWNLOADS_DIR / 'evasoes_intervias'

    PATH_OUT = ensure_parent(RESULTS_EVASAO_DIR / 'resultado_intervias.xlsx')

    validador = ValidadorPDFNovoFormato(
        yolo_weights_path=PATH_YOLO,
        usar_gpu=True,
    )

    validador.processar(
        PASTA_PDFS,
        PATH_OUT,
    )



