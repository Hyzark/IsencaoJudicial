"""
ValidadorExcelEixoVP
====================
Processa planilhas EixoVP reutilizando o pipeline de YOLO + EasyOCR do
app_excel_spvias.py.

Colunas esperadas:
  - PLACA
  - ID_PASSAGEM
  - DATA_HORA

Imagens:
  - tenta abrir caminhos/URLs em colunas com nome contendo IMAGEM ou FOTO;
  - se as celulas estiverem vazias, busca imagens embutidas no .xlsx em
    xl/media, usando xl/drawings para associar a imagem a linha da passagem.

Criterio OCR:
  - aprova se o OCR encontrar pelo menos 4 caracteres em comum com a placa
    esperada, mesmo quando a distancia de Levenshtein seria alta.
"""

from __future__ import annotations

import os
import posixpath
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from glob import glob
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd
from PIL import Image
from tqdm import tqdm


PROJETO_DIR = Path(__file__).resolve().parent
if str(PROJETO_DIR) not in sys.path:
    sys.path.insert(0, str(PROJETO_DIR))

from config import DOWNLOADS_DIR, MODEL_PATH, RESULTS_EIXO_DIR, ensure_parent
from app_excel_spvias import ValidadorExcelPro


NAO_IDENTIFICADO = 'Nao foi possivel identificar os caracteres'
EMU_POR_PONTO = 12_700

NS_XLSX = {
    'xdr': 'http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'rel': 'http://schemas.openxmlformats.org/package/2006/relationships',
}


class ValidadorExcelEixoVP(ValidadorExcelPro):
    """
    Leitor especifico EixoVP.

    A classe herda todo o pipeline pesado do SPVias e adapta apenas:
      - leitura das colunas da planilha;
      - busca da primeira imagem valida por linha;
      - criterio de aprovacao por 4 caracteres coincidentes.
    """

    def _normalizar_nome_coluna(self, nome: object) -> str:
        texto = str(nome).strip().lower()
        return re.sub(r'[^a-z0-9]+', '', texto)

    def _resolver_coluna(self, df: pd.DataFrame, candidatos: list[str]) -> str | None:
        mapa = {self._normalizar_nome_coluna(col): col for col in df.columns}
        for candidato in candidatos:
            coluna = mapa.get(self._normalizar_nome_coluna(candidato))
            if coluna is not None:
                return coluna
        return None

    def _detectar_linha_cabecalho(self, caminho_excel: str) -> int:
        previa = pd.read_excel(caminho_excel, header=None, nrows=40, dtype=object)
        obrigatorias = {
            self._normalizar_nome_coluna('PLACA'),
            self._normalizar_nome_coluna('ID_PASSAGEM'),
            self._normalizar_nome_coluna('DATA_HORA'),
        }

        for idx, row in previa.iterrows():
            nomes = {
                self._normalizar_nome_coluna(valor)
                for valor in row.tolist()
                if self._valor_valido(valor)
            }
            if obrigatorias.issubset(nomes):
                return int(idx)

        raise ValueError('Cabecalho com PLACA, ID_PASSAGEM e DATA_HORA nao encontrado.')

    def _colunas_imagem(self, df: pd.DataFrame) -> list[str]:
        colunas = []
        for col in df.columns:
            nome = self._normalizar_nome_coluna(col)
            if 'imagem' in nome or 'foto' in nome:
                colunas.append(col)
        return colunas

    def _valor_valido(self, valor: object) -> bool:
        if pd.isna(valor):
            return False
        texto = str(valor).strip()
        return bool(texto) and texto.lower() not in {'nan', 'none', 'null'}

    def _abrir_imagem_de_valor(self, valor: object, pasta_base: str) -> bytes | None:
        if not self._valor_valido(valor):
            return None

        texto = str(valor).strip().strip('"')
        parsed = urlparse(texto)

        try:
            if parsed.scheme in {'http', 'https'}:
                req = Request(texto, headers={'User-Agent': 'Mozilla/5.0'})
                with urlopen(req, timeout=20) as resp:
                    dados = resp.read()
            else:
                caminho = Path(texto)
                if not caminho.is_absolute():
                    caminho = Path(pasta_base) / caminho
                if not caminho.exists() or not caminho.is_file():
                    return None
                dados = caminho.read_bytes()

            with Image.open(BytesIO(dados)) as img:
                img.verify()
            return dados
        except Exception as e:
            print(f'  AVISO: imagem invalida/indisponivel ({texto}): {e}')
            return None

    def _primeira_imagem_linha(
        self,
        row: pd.Series,
        colunas_imagem: list[str],
        pasta_base: str,
        imagens_embutidas: dict[int, list[tuple[bytes, str]]] | None,
        linha_excel: int,
    ) -> tuple[list[bytes], str]:
        for coluna in colunas_imagem:
            img_bytes = self._abrir_imagem_de_valor(row.get(coluna), pasta_base)
            if img_bytes:
                return [img_bytes], str(coluna)

        if imagens_embutidas:
            imagens_linha = imagens_embutidas.get(linha_excel, [])
            if imagens_linha:
                img_bytes, origem = imagens_linha[0]
                return [img_bytes], origem

        return [], ''

    def _mapear_imagens_embutidas(self, caminho_excel: str) -> dict[int, list[tuple[bytes, str]]]:
        mapa: dict[int, list[tuple[bytes, str]]] = {}

        with zipfile.ZipFile(caminho_excel, 'r') as z:
            arquivos = set(z.namelist())
            sheet_path = 'xl/worksheets/sheet1.xml'
            rels_sheet = 'xl/worksheets/_rels/sheet1.xml.rels'
            if sheet_path not in arquivos or rels_sheet not in arquivos:
                return mapa

            alturas_linha = self._ler_alturas_linhas_emu(z, sheet_path)
            drawing_paths = self._drawing_paths_da_sheet(z, rels_sheet)

            for drawing_path in drawing_paths:
                if drawing_path not in arquivos:
                    continue

                drawing_rels_path = self._rels_path_do_drawing(drawing_path)
                if drawing_rels_path not in arquivos:
                    continue

                rid_para_media = self._rid_para_media(z, drawing_rels_path)
                drawing_root = ET.fromstring(z.read(drawing_path))
                anchors = (
                    list(drawing_root.findall('xdr:twoCellAnchor', NS_XLSX))
                    + list(drawing_root.findall('xdr:oneCellAnchor', NS_XLSX))
                )

                for anchor in anchors:
                    from_elem = anchor.find('xdr:from', NS_XLSX)
                    if from_elem is None:
                        continue

                    row = int(from_elem.find('xdr:row', NS_XLSX).text) + 1
                    row_off = int(from_elem.find('xdr:rowOff', NS_XLSX).text or 0)

                    altura_linha = alturas_linha.get(row, 15 * EMU_POR_PONTO)
                    if row_off > altura_linha * 0.5:
                        row += 1

                    pic = anchor.find('xdr:pic', NS_XLSX)
                    if pic is None:
                        continue

                    blip = pic.find('.//a:blip', NS_XLSX)
                    if blip is None:
                        continue

                    rid = blip.get(
                        '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed'
                    )
                    media_path = rid_para_media.get(rid)
                    if not media_path or media_path not in arquivos:
                        continue

                    mapa.setdefault(row, []).append((z.read(media_path), media_path))

        return mapa

    def _ler_alturas_linhas_emu(self, z: zipfile.ZipFile, sheet_path: str) -> dict[int, int]:
        alturas = {}
        root = ET.fromstring(z.read(sheet_path))
        ns_sheet = {'ws': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        for row in root.findall('.//ws:row', ns_sheet):
            linha = row.get('r')
            altura = row.get('ht')
            if linha and altura:
                alturas[int(linha)] = int(float(altura) * EMU_POR_PONTO)
        return alturas

    def _drawing_paths_da_sheet(self, z: zipfile.ZipFile, rels_sheet: str) -> list[str]:
        root = ET.fromstring(z.read(rels_sheet))
        paths = []
        for rel in root.findall('rel:Relationship', NS_XLSX):
            if 'drawing' not in rel.get('Type', '').lower():
                continue
            paths.append(self._normalizar_zip_path('xl/worksheets', rel.get('Target', '')))
        return paths

    def _rels_path_do_drawing(self, drawing_path: str) -> str:
        pasta, nome = drawing_path.rsplit('/', 1)
        return f'{pasta}/_rels/{nome}.rels'

    def _rid_para_media(self, z: zipfile.ZipFile, drawing_rels_path: str) -> dict[str, str]:
        base_dir = drawing_rels_path.rsplit('/_rels/', 1)[0]
        root = ET.fromstring(z.read(drawing_rels_path))
        mapa = {}
        for rel in root.findall('rel:Relationship', NS_XLSX):
            rid = rel.get('Id')
            target = rel.get('Target', '')
            if rid:
                mapa[rid] = self._normalizar_zip_path(base_dir, target)
        return mapa

    def _normalizar_zip_path(self, base_dir: str, target: str) -> str:
        if target.startswith('/'):
            return target.lstrip('/')
        return posixpath.normpath(posixpath.join(base_dir, target))

    def _lcs_len(self, a: str, b: str) -> int:
        anterior = [0] * (len(b) + 1)
        for char_a in a:
            atual = [0]
            for j, char_b in enumerate(b, start=1):
                if char_a == char_b:
                    atual.append(anterior[j - 1] + 1)
                else:
                    atual.append(max(anterior[j], atual[j - 1]))
            anterior = atual
        return anterior[-1]

    def _contar_caracteres_coincidentes(self, placa_esperada: str, placa_ocr: str) -> int:
        esperada = self._limpar_texto_ocr(str(placa_esperada))
        ocr = self._limpar_texto_ocr(str(placa_ocr))

        if not esperada or not ocr or ocr == self._limpar_texto_ocr(NAO_IDENTIFICADO):
            return 0

        por_posicao = sum(1 for a, b in zip(esperada, ocr) if a == b)
        lcs = self._lcs_len(esperada, ocr)
        comuns = sum((Counter(esperada) & Counter(ocr)).values())
        return max(por_posicao, lcs, comuns)

    def _avaliar_bloco(self, placa_esperada: str, imagens_bytes: list[bytes]) -> dict:
        if not imagens_bytes:
            return {
                'placa_ocr': NAO_IDENTIFICADO,
                'ocr_bruto': '',
                'status': 'Revisar - sem imagem',
                'diferenca': 999,
                'imagem_usada': '',
                'total_imgs': 0,
                'imgs_com_ocr': 0,
                'coincidencias': 0,
            }

        mascara_esperada = self._descobrir_mascara(placa_esperada)
        placa_texto_limpa = self._normalizar_placa_ocr(placa_esperada, mascara_esperada)

        melhor_placa = ''
        melhor_distancia = 999
        melhor_indice = None
        melhor_ocr_bruto = ''
        melhor_coincidencias = 0
        imgs_com_ocr = 0

        for indice, img_bytes in enumerate(imagens_bytes):
            ocr_bruto = self._processar_imagem_yolo_ocr(img_bytes, mascara_esperada)
            if ocr_bruto == NAO_IDENTIFICADO or not ocr_bruto:
                continue

            imgs_com_ocr += 1
            placa_ocr = self._normalizar_placa_ocr(ocr_bruto, mascara_esperada)
            distancia = self._distancia_placas(placa_texto_limpa, placa_ocr)
            coincidencias = self._contar_caracteres_coincidentes(
                placa_texto_limpa,
                placa_ocr,
            )

            if coincidencias > melhor_coincidencias or (
                coincidencias == melhor_coincidencias and distancia < melhor_distancia
            ):
                melhor_coincidencias = coincidencias
                melhor_distancia = distancia
                melhor_placa = placa_ocr
                melhor_indice = indice
                melhor_ocr_bruto = ocr_bruto

            if coincidencias >= 4:
                break

        if melhor_coincidencias >= 4:
            status = 'Aprovado - busca nas imagens'
            placa_final = placa_texto_limpa
        else:
            status = 'Revisar'
            placa_final = melhor_placa or NAO_IDENTIFICADO

        return {
            'placa_ocr': placa_final,
            'ocr_bruto': melhor_ocr_bruto,
            'status': status,
            'diferenca': round(melhor_distancia, 2),
            'imagem_usada': (melhor_indice + 1) if melhor_indice is not None else '',
            'total_imgs': len(imagens_bytes),
            'imgs_com_ocr': imgs_com_ocr,
            'coincidencias': melhor_coincidencias,
        }

    def _formatar_valor(self, valor: object) -> object:
        if pd.isna(valor):
            return ''
        return valor

    def processar(self, entrada_excel: str, caminho_saida: str):
        caminho_entrada = Path(entrada_excel)
        if caminho_entrada.is_file():
            arquivos_excel = [str(caminho_entrada)]
        else:
            arquivos_excel = glob(os.path.join(entrada_excel, '*.xlsx'))

        if not arquivos_excel:
            print('Nenhum arquivo .xlsx encontrado.')
            return

        registros = []

        for caminho_excel in arquivos_excel:
            print(f'\nProcessando: {caminho_excel}')
            pasta_base = str(Path(caminho_excel).parent)

            print('Detectando cabecalho...')
            try:
                linha_cabecalho = self._detectar_linha_cabecalho(caminho_excel)
            except Exception as e:
                print(f'  Pulando arquivo - {e}')
                continue

            print(f'  Cabecalho encontrado na linha {linha_cabecalho + 1}.')
            df = pd.read_excel(caminho_excel, header=linha_cabecalho)
            df = df.dropna(how='all')

            col_placa = self._resolver_coluna(df, ['PLACA'])
            col_id = self._resolver_coluna(df, ['ID_PASSAGEM'])
            col_data_hora = self._resolver_coluna(df, ['DATA_HORA'])
            colunas_imagem = self._colunas_imagem(df)

            faltantes = [
                nome for nome, coluna in {
                    'PLACA': col_placa,
                    'ID_PASSAGEM': col_id,
                    'DATA_HORA': col_data_hora,
                }.items()
                if coluna is None
            ]
            if faltantes:
                print(f'  Pulando arquivo - colunas obrigatorias ausentes: {", ".join(faltantes)}')
                continue

            imagens_embutidas = self._mapear_imagens_embutidas(caminho_excel)

            print(f'  {len(df)} passagens encontradas.')
            print(f'  Colunas de imagem/caminho: {len(colunas_imagem)}')
            print(f'  Linhas com imagem embutida: {len(imagens_embutidas)}')
            print('Iniciando analise YOLO + OCR...')

            for idx, row in tqdm(df.iterrows(), total=len(df)):
                linha_excel = int(idx) + linha_cabecalho + 2
                placa = str(row.get(col_placa, '')).strip().upper()
                imagens, origem_imagem = self._primeira_imagem_linha(
                    row,
                    colunas_imagem,
                    pasta_base,
                    imagens_embutidas,
                    linha_excel,
                )
                resultado = self._avaliar_bloco(placa, imagens)

                registros.append({
                    'Arquivo': os.path.basename(caminho_excel),
                    'Linha Excel': linha_excel,
                    'ID_PASSAGEM': self._formatar_valor(row.get(col_id)),
                    'DATA_HORA': self._formatar_valor(row.get(col_data_hora)),
                    'Placa Esperada': placa,
                    'Origem Imagem': origem_imagem,
                    'Placa OCR': resultado['placa_ocr'],
                    'OCR Bruto': resultado['ocr_bruto'],
                    'Status OCR': resultado['status'],
                    'Caracteres coincidentes': resultado['coincidencias'],
                    'Diferenca OCR': resultado['diferenca'],
                    'Imagem Usada': resultado['imagem_usada'],
                    'Total Imagens': resultado['total_imgs'],
                    'Imagens c/ OCR': resultado['imgs_com_ocr'],
                })

        if not registros:
            print('\nNenhum registro gerado. Verifique os arquivos de entrada.')
            return

        df_saida = pd.DataFrame(registros)
        df_saida.to_excel(caminho_saida, index=False)

        total = len(df_saida)
        aprovados = (df_saida['Status OCR'] == 'Aprovado - busca nas imagens').sum()
        revisar = (df_saida['Status OCR'] == 'Revisar').sum()
        sem_img = (df_saida['Status OCR'] == 'Revisar - sem imagem').sum()

        print(f"\n{'='*54}")
        print(f'  Resultado final: {total} passagens')
        print(f'  Aprovado:        {aprovados:3d} ({aprovados/total*100:.0f}%)')
        print(f'  Revisar:         {revisar:3d} ({revisar/total*100:.0f}%)')
        print(f'  Sem imagem:      {sem_img:3d} ({sem_img/total*100:.0f}%)')
        print(f"{'='*54}")
        print(f'  Arquivo gerado: {caminho_saida}')


if __name__ == '__main__':
    PATH_YOLO = MODEL_PATH

    ENTRADA_EXCEL = DOWNLOADS_DIR / 'eixo_vp'

    PATH_OUT = ensure_parent(RESULTS_EIXO_DIR / 'resultado_eixovp.xlsx')

    validador = ValidadorExcelEixoVP(
        yolo_weights_path=PATH_YOLO,
        usar_gpu=True,
    )

    validador.processar(
        ENTRADA_EXCEL,
        PATH_OUT,
    )





