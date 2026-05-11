"""
ValidadorImgTamoios
===================
Processa imagens físicas de evasão/isenção da Tamoios, cruzando os dados
com uma planilha de referência Excel.

Fluxo
-----
  1. Lê a planilha de referência (a partir da linha 5).
  2. Para cada imagem na pasta de entrada:
       a. Extrai o ID do nome do arquivo (tudo antes do primeiro hífen).
       b. Busca o ID na planilha → obtém Placa, Data, Hora e Categoria.
       c. Roda YOLO + OCR na imagem para ler a placa.
       d. Compara placa da planilha × placa OCR via Levenshtein customizado.
       e. Decide status: Aprovado / Revisar / ID não encontrado / ID fora do padrão.
  3. Gera relatório Excel consolidado.

Herança
-------
Toda a lógica pesada (YOLO, OCR, pré-processamento, Levenshtein, normalização)
é herdada de ValidadorExcelPro sem nenhuma alteração.

Dependências
------------
  pip install openpyxl pandas pillow tqdm
  (ultralytics, easyocr, opencv já são dependências do projeto)
"""

from __future__ import annotations

import os
import re
from glob import glob
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from app_excel_spvias import ValidadorExcelPro
from config import DOWNLOADS_DIR, MODEL_PATH, RESULTS_ISENCAO_DIR, ensure_parent

# Extensões de imagem aceitas
_EXTENSOES_IMG = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# Regex do ID: sequência alfanumérica antes do primeiro hífen
# Ex: "060113SMA202401240617220612-F02.jpg" → "060113SMA202401240617220612"
_RE_ID_ARQUIVO = re.compile(r'^([A-Za-z0-9]+)-')


class ValidadorImgTamoios(ValidadorExcelPro):
    """
    Subclasse especializada em imagens físicas da Tamoios.

    Herda e reutiliza sem alteração:
      _processar_imagem_yolo_ocr  · _avaliar_bloco
      _normalizar_placa_ocr       · _distancia_placas · _pre_processar_imagem
      _limpar_texto_ocr           · _corrigir_por_mascara · _descobrir_mascara
    """

    # ── Leitura da planilha de referência ──────────────────────────────────────

    def _carregar_planilha(self, caminho_excel: str) -> pd.DataFrame:
        """
        Lê a planilha de referência a partir da linha 5 (header na linha 5,
        dados a partir da linha 6).

        Tenta localizar automaticamente as colunas ID, Placa, Data, Hora e
        Categoria por nome (case-insensitive), tornando o código robusto a
        pequenas variações de layout.
        """
        # header=4 → Python lê a linha 5 do Excel como cabeçalho (0-based)
        df = pd.read_excel(caminho_excel, header=4, dtype=str)

        # Normaliza nomes de colunas: remove espaços e acentos, caixa baixa
        df.columns = [self._normalizar_coluna(c) for c in df.columns]

        # Mapa de sinônimos aceitos para cada campo
        sinonimos = {
            'id':        ['id', 'passagem', 'numero', 'numeroid', 'idpassagem',
                          'idsistema', 'id sistema', 'id passagem', 'ID Transação', 'id transacao'],
            'placa':     ['placa', 'placaveiculo', 'placa veiculo'],
            'data':      ['data', 'data passagem', 'datapassagem'],
            'hora':      ['hora', 'hora passagem', 'horapassagem'],
            'categoria': ['categoria', 'cat', 'classeveiculo', 'classe veiculo',
                          'tipoveiculo'],
        }

        mapa = {}   # campo_interno → nome_real_no_df
        for campo, nomes in sinonimos.items():
            for nome in nomes:
                if nome in df.columns:
                    mapa[campo] = nome
                    break

        colunas_ausentes = [c for c in sinonimos if c not in mapa]
        if colunas_ausentes:
            print(f'AVISO: colunas não encontradas na planilha: {colunas_ausentes}')
            print(f'  Colunas disponíveis: {list(df.columns)}')

        # Renomeia para nomes canônicos internos
        df = df.rename(columns={v: k for k, v in mapa.items()})

        # Garante que todas as colunas canônicas existam (mesmo que vazias)
        for campo in sinonimos:
            if campo not in df.columns:
                df[campo] = ''

        # Remove linhas completamente vazias
        df = df.dropna(how='all').reset_index(drop=True)

        # Normaliza a coluna ID: strip + uppercase
        df['id'] = df['id'].fillna('').str.strip().str.upper()

        return df

    def _normalizar_coluna(self, nome: str) -> str:
        """Remove espaços extras e coloca em minúsculas para busca case-insensitive."""
        import unicodedata
        nome = str(nome).strip().lower()
        nome = unicodedata.normalize('NFKD', nome)
        nome = ''.join(c for c in nome if not unicodedata.combining(c))
        return re.sub(r'\s+', ' ', nome)

    # ── Extração do ID a partir do nome do arquivo ─────────────────────────────

    def _extrair_id_arquivo(self, nome_arquivo: str) -> tuple[str, str]:
        """
        Extrai o ID do nome do arquivo.

        Retorna (id, status_id) onde status_id é:
          'OK'               → ID extraído com sucesso (havia hífen)
          'ID fora do padrão' → nome sem hífen (não segue a convenção esperada)
        """
        stem = Path(nome_arquivo).stem   # sem extensão
        m = _RE_ID_ARQUIVO.match(stem)
        if m:
            return m.group(1).upper(), 'OK'
        # Sem hífen: considera o nome inteiro como ID mas marca como fora do padrão
        return stem.upper(), 'ID fora do padrão'

    # ── Leitura dos bytes da imagem ────────────────────────────────────────────

    def _ler_imagem_bytes(self, caminho_img: str) -> bytes | None:
        try:
            with open(caminho_img, 'rb') as f:
                return f.read()
        except Exception as e:
            print(f'  Erro ao ler imagem {caminho_img}: {e}')
            return None

    # ── Pipeline principal ─────────────────────────────────────────────────────

    def processar(self, caminho_excel: str, pasta_imagens: str, caminho_saida: str):
        """
        Pipeline completo para imagens físicas da Tamoios.

        1. Carrega planilha de referência.
        2. Lista imagens da pasta.
        3. Para cada imagem: extrai ID → cruza planilha → YOLO+OCR → decide status.
        4. Consolida e salva Excel de saída.
        5. Cria uma segunda aba somente com o melhor resultado de cada ID.
        """

        # ── Função auxiliar para ranking dos resultados ──────────────────────────
        def _score_linha(row):
            """
            Quanto MENOR o score, melhor o resultado.
            Prioriza:
            1. Menor diferença OCR
            2. Aprovados
            3. IDs válidos
            """

            status_ocr = str(row.get('Status OCR', ''))
            status_id = str(row.get('Status ID', ''))
            diferenca = row.get('Diferença OCR', 999)

            try:
                diferenca = int(diferenca)
            except:
                diferenca = 999

            # Penalidade baseada no status OCR
            if status_ocr == 'Aprovado - busca nas imagens':
                peso_status = 0
            elif status_ocr == 'Revisar':
                peso_status = 100
            elif status_ocr == 'Revisar - sem imagem':
                peso_status = 200
            else:
                peso_status = 300

            # Penalidade para IDs inválidos
            if status_id != 'ID - OK':
                peso_status += 1000

            return peso_status + diferenca

        # ── Planilha de referência ──
        print(f'Carregando planilha de referência: {caminho_excel}')

        try:
            df_ref = self._carregar_planilha(caminho_excel)
        except Exception as e:
            print(f'Erro ao ler planilha: {e}')
            return

        print(f'  {len(df_ref)} registros carregados da planilha.')

        # Índice ID → linha para busca O(1)
        indice_id = {
            row['id']: row
            for _, row in df_ref.iterrows()
            if row['id']
        }

        # ── Imagens ──
        imagens = [
            p for p in Path(pasta_imagens).iterdir()
            if p.is_file() and p.suffix.lower() in _EXTENSOES_IMG
        ]

        imagens.sort()

        if not imagens:
            print(f'Nenhuma imagem encontrada em: {pasta_imagens}')
            return

        print(f'  {len(imagens)} imagem(ns) encontrada(s) na pasta.')

        # ── Processamento ──
        registros = []

        for caminho_img in tqdm(imagens, desc='Processando imagens'):

            nome_arq = caminho_img.name

            # 1. Extrai ID do nome do arquivo
            id_arquivo, status_id = self._extrair_id_arquivo(nome_arq)

            # 2. Cruza com a planilha
            linha_ref = indice_id.get(id_arquivo)

            if status_id == 'ID fora do padrão':

                registros.append({
                    'Arquivo':         nome_arq,
                    'ID Extraído':     id_arquivo,
                    'Status ID':       'ID fora do padrão',
                    'Placa Planilha':  '',
                    'Data':            '',
                    'Hora':            '',
                    'Categoria':       '',
                    'Placa OCR':       '',
                    'OCR Bruto':       '',
                    'Status OCR':      '',
                    'Diferença OCR':   '',
                    'Imagem Usada':    '',
                    'Total Imagens':   1,
                    'Imagens c/ OCR':  '',
                })

                continue

            if linha_ref is None:

                registros.append({
                    'Arquivo':         nome_arq,
                    'ID Extraído':     id_arquivo,
                    'Status ID':       'ID não encontrado na planilha',
                    'Placa Planilha':  '',
                    'Data':            '',
                    'Hora':            '',
                    'Categoria':       '',
                    'Placa OCR':       '',
                    'OCR Bruto':       '',
                    'Status OCR':      '',
                    'Diferença OCR':   '',
                    'Imagem Usada':    '',
                    'Total Imagens':   1,
                    'Imagens c/ OCR':  '',
                })

                continue

            # ID encontrado
            placa_planilha = str(linha_ref.get('placa', '')).strip().upper()
            data           = str(linha_ref.get('data',  '')).strip()
            hora           = str(linha_ref.get('hora',  '')).strip()
            categoria      = str(linha_ref.get('categoria', '')).strip()

            # 3. Lê imagem e roda YOLO + OCR
            img_bytes = self._ler_imagem_bytes(str(caminho_img))

            resultado = self._avaliar_bloco(
                placa_planilha,
                [img_bytes] if img_bytes else []
            )

            registros.append({
                'Arquivo':         nome_arq,
                'ID Extraído':     id_arquivo,
                'Status ID':       'ID - OK',
                'Placa Planilha':  placa_planilha,
                'Data':            data,
                'Hora':            hora,
                'Categoria':       categoria,
                'Placa OCR':       resultado['placa_ocr'],
                'OCR Bruto':       resultado['ocr_bruto'],
                'Status OCR':      resultado['status'],
                'Diferença OCR':   resultado['diferenca'],
                'Imagem Usada':    resultado['imagem_usada'],
                'Total Imagens':   resultado['total_imgs'],
                'Imagens c/ OCR':  resultado['imgs_com_ocr'],
            })

        if not registros:
            print('Nenhum registro gerado.')
            return

        # ── DataFrame principal ──
        df = pd.DataFrame(registros)

        # ── Segunda aba: melhor resultado por ID ────────────────────────────────
        df_melhores = df.copy()

        # Cria score de prioridade
        df_melhores['_score'] = df_melhores.apply(_score_linha, axis=1)

        # Ordena do melhor para o pior
        df_melhores = df_melhores.sort_values(
            by=['_score', 'Diferença OCR']
        )

        # Mantém somente o melhor resultado de cada ID
        df_melhores = df_melhores.drop_duplicates(
            subset=['ID Extraído'],
            keep='first'
        )

        # Remove coluna auxiliar
        df_melhores = df_melhores.drop(columns=['_score'])

        # ── Salva Excel com 2 abas ──────────────────────────────────────────────
        with pd.ExcelWriter(caminho_saida, engine='openpyxl') as writer:

            # Aba original
            df.to_excel(
                writer,
                sheet_name='Resultado Completo',
                index=False
            )

            # Aba filtrada
            df_melhores.to_excel(
                writer,
                sheet_name='Melhor Resultado ID',
                index=False
            )

        # ── Resumo ──
        total        = len(df)
        id_ok        = (df['Status ID'] == 'ID - OK').sum()
        id_nao_enc   = (df['Status ID'] == 'ID não encontrado na planilha').sum()
        id_fora      = (df['Status ID'] == 'ID fora do padrão').sum()
        aprovados    = (df['Status OCR'] == 'Aprovado - busca nas imagens').sum()
        revisar      = (df['Status OCR'] == 'Revisar').sum()
        sem_img      = (df['Status OCR'] == 'Revisar - sem imagem').sum()

        print(f"\n{'='*54}")
        print(f"  Resultado final: {total} imagens")
        print(f"  ID OK:                {id_ok:3d} ({id_ok/total*100:.0f}%)")
        print(f"  ID não encontrado:    {id_nao_enc:3d} ({id_nao_enc/total*100:.0f}%)")
        print(f"  ID fora do padrão:    {id_fora:3d} ({id_fora/total*100:.0f}%)")
        print(f"  ─────────────────────────────────────────")
        print(f"  Aprovado OCR:         {aprovados:3d} ({aprovados/total*100:.0f}%)")
        print(f"  Revisar OCR:          {revisar:3d} ({revisar/total*100:.0f}%)")
        print(f"  Sem imagem:           {sem_img:3d} ({sem_img/total*100:.0f}%)")
        print(f"{'='*54}")
        print(f"  Arquivo gerado: {caminho_saida}")
        print(f"  Aba extra criada: 'Melhor Resultado ID'")


# ── Ponto de entrada ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    PATH_YOLO      = MODEL_PATH
    PLANILHA_REF   = DOWNLOADS_DIR / 'tamoios_referencia.xlsx'
    PASTA_IMAGENS  = DOWNLOADS_DIR / 'isencao_tamoios/P1'
    PATH_OUT       = ensure_parent(RESULTS_ISENCAO_DIR / 'resultado_tamoios.xlsx')

    validador = ValidadorImgTamoios(
        yolo_weights_path=str(PATH_YOLO),
        usar_gpu=True,
    )

    validador.processar(
        caminho_excel=str(PLANILHA_REF),
        pasta_imagens=str(PASTA_IMAGENS),
        caminho_saida=str(PATH_OUT),
    )