"""
ValidadorPDFEcovias
===================
Processa PDFs de evasao no layout Ecovias, reaproveitando o pipeline de
YOLO + OCR + validacao de placas existente em app_excel.py.

O arquivo le:
  - a linha de passagem do PDF;
  - as fotos reais embutidas no PDF;
  - valida campos essenciais da passagem;
  - compara a placa do texto contra a placa lida por OCR nas imagens.

Dependencias:
  pip install pymupdf pdfplumber
"""

from __future__ import annotations

import os
import re
import unicodedata
from glob import glob
from datetime import datetime

import fitz
import pandas as pd
import pdfplumber

from app_excel_spvias import ValidadorExcelPro
from config import DOWNLOADS_DIR, MODEL_PATH, RESULTS_EVASAO_DIR, ensure_parent


# Imagens menores que isso costumam ser logo, assinatura ou carimbo.
_MIN_AREA_FOTO_PX = 100_000

_SENTIDOS = (
    'NORTE',
    'SUL',
    'LESTE',
    'OESTE',
    'INTERIOR',
    'CAPITAL',
    'CRESCENTE',
    'DECRESCENTE',
)


class ValidadorPDFEcovias(ValidadorExcelPro):
    """
    Subclasse especializada em PDF.

    Herda do app_excel.py toda a parte pesada:
      - YOLO
      - EasyOCR
      - normalizacao de placa
      - distancia de Levenshtein customizada
      - decisao final de aprovado/revisar
    """

    def _normalizar_texto_pdf(self, texto: str) -> str:
        """Uniformiza espacos e hifens que costumam vir diferentes no PDF."""
        texto = texto.replace('\xa0', ' ')
        texto = texto.replace('\u2010', '-')
        texto = texto.replace('\u2011', '-')
        texto = texto.replace('\u2012', '-')
        texto = texto.replace('\u2013', '-')
        texto = texto.replace('\u2014', '-')
        texto = texto.replace('\u2212', '-')
        return re.sub(r'[ \t]+', ' ', texto).strip()

    def _extrair_texto_pdf(self, caminho_pdf: str) -> str:
        textos = []

        with pdfplumber.open(caminho_pdf) as pdf:
            for page in pdf.pages:
                texto = page.extract_text(x_tolerance=1, y_tolerance=3) or ''
                if texto:
                    textos.append(texto)

        return self._normalizar_texto_pdf('\n'.join(textos))

    def _encontrar_linha_passagem(self, texto: str) -> str | None:
        for linha in texto.splitlines():
            linha = self._normalizar_texto_pdf(linha)
            if (
                re.search(r'\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}', linha)
                and self.regex_placa.search(linha.upper())
            ):
                return linha

        # Fallback para PDFs em que a extracao quebra as linhas da tabela.
        texto_uma_linha = self._normalizar_texto_pdf(texto.replace('\n', ' '))
        match = re.search(
            r'(.{0,80}\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}.+?'
            r'[A-Z]{3}\d[A-Z\d]\d{2}.*?)(?:\s+CONCESSIONARIA|\s+COMUNICACAO|\s+DADOS\s+DO|\Z)',
            self._remover_acentos_simples(texto_uma_linha),
            flags=re.IGNORECASE,
        )
        if match:
            return self._normalizar_texto_pdf(match.group(1))

        return None

    def _remover_acentos_simples(self, texto: str) -> str:
        normalizado = unicodedata.normalize('NFKD', texto)
        return ''.join(char for char in normalizado if not unicodedata.combining(char))

    def _parse_prefixo_antes_placa(self, prefixo: str) -> dict:
        sentido_regex = '|'.join(_SENTIDOS)
        match = re.match(
            rf'^(?P<praca>.+?)\s+(?P<pista>\d+)\s+(?P<sentido>{sentido_regex})\s+(?P<municipio>.+?)\s+(?P<codigo>\d+)$',
            prefixo,
            flags=re.IGNORECASE,
        )

        if match:
            dados = match.groupdict()
            return {k: self._normalizar_texto_pdf(v).upper() for k, v in dados.items()}

        tokens = prefixo.split()
        return {
            'praca': prefixo,
            'pista': '',
            'sentido': '',
            'municipio': '',
            'codigo': tokens[-1] if tokens and tokens[-1].isdigit() else '',
        }

    def _parse_sufixo_depois_placa(self, sufixo: str) -> dict:
        # Ecovias costuma vir como: Ford - KA 1 R$ 9,70 Envio para Autuacao Sim
        match_ecovias = re.match(
            r'^(?P<modelo>.+?)\s+(?P<eixos>\d+)\s+R\$\s*'
            r'(?P<tarifa>\d+(?:[.,]\d{2})?)\s*(?P<responsavel>.*)$',
            sufixo,
            flags=re.IGNORECASE,
        )

        if match_ecovias:
            dados = match_ecovias.groupdict()
            return {
                'modelo': self._normalizar_texto_pdf(dados['modelo']),
                'eixos': dados['eixos'],
                'tarifa': dados['tarifa'].replace('.', ','),
                'responsavel': self._normalizar_texto_pdf(dados['responsavel']),
            }

        match = re.match(
            r'^(?P<modelo>.+?)\s+(?P<eixos>\d+)\s+(?P<tarifa>\d+(?:[.,]\d{2})?)\s+R\$\s*(?P<responsavel>.+)$',
            sufixo,
            flags=re.IGNORECASE,
        )

        if match:
            dados = match.groupdict()
            return {
                'modelo': self._normalizar_texto_pdf(dados['modelo']),
                'eixos': dados['eixos'],
                'tarifa': dados['tarifa'].replace('.', ','),
                'responsavel': self._normalizar_texto_pdf(dados['responsavel']),
            }

        match_sem_responsavel = re.match(
            r'^(?P<modelo>.+?)\s+(?P<eixos>\d+)\s+(?P<tarifa>\d+(?:[.,]\d{2})?)\s+R\$?$',
            sufixo,
            flags=re.IGNORECASE,
        )

        if match_sem_responsavel:
            dados = match_sem_responsavel.groupdict()
            return {
                'modelo': self._normalizar_texto_pdf(dados['modelo']),
                'eixos': dados['eixos'],
                'tarifa': dados['tarifa'].replace('.', ','),
                'responsavel': '',
            }

        return {
            'modelo': sufixo,
            'eixos': '',
            'tarifa': '',
            'responsavel': '',
        }

    def _ler_passagem_pdf(self, caminho_pdf: str) -> dict | None:
        texto = self._extrair_texto_pdf(caminho_pdf)

        if not texto:
            print(f'AVISO: texto nao encontrado em {caminho_pdf}')
            return None

        linha = self._encontrar_linha_passagem(texto)
        if not linha:
            print('AVISO: linha da passagem nao encontrada.')
            return None

        linha = self._normalizar_texto_pdf(linha)
        match_base = re.match(
            r'^(?P<id>.*?)\s*'
            r'(?P<data>\d{2}/\d{2}/\d{4})\s+'
            r'(?P<hora>\d{2}:\d{2}:\d{2})\s+'
            r'(?P<resto>.+)$',
            linha,
            flags=re.IGNORECASE,
        )

        if not match_base:
            print('Erro ao interpretar dados minimos da linha:')
            print(linha)
            return None

        dados = match_base.groupdict()
        resto = dados['resto']

        match_placa = self.regex_placa.search(resto.upper())
        if not match_placa:
            print('Erro: placa nao encontrada na linha:')
            print(linha)
            return None

        placa = match_placa.group(1).upper()
        prefixo = self._normalizar_texto_pdf(resto[:match_placa.start()])
        sufixo = self._normalizar_texto_pdf(resto[match_placa.end():])

        sp, km, prefixo_passagem = self._extrair_sp_km_do_prefixo(prefixo)
        dados_prefixo = self._parse_prefixo_antes_placa(prefixo_passagem or prefixo)
        dados_sufixo = self._parse_sufixo_depois_placa(sufixo)

        passagem = {
            'id': self._normalizar_texto_pdf(dados['id']).upper(),
            'data': dados['data'],
            'hora': dados['hora'],
            'sp': sp,
            'km': km,
            'placa': placa,
            'linha_bruta': linha,
            **dados_prefixo,
            **dados_sufixo,
        }

        status_passagem, alertas = self._validar_passagem(passagem)
        passagem['status_passagem'] = status_passagem
        passagem['alertas_passagem'] = '; '.join(alertas)

        return passagem

    def _extrair_sp_km_do_prefixo(self, prefixo: str) -> tuple[str, str, str]:
        tokens = prefixo.split()
        if not tokens:
            return '', '', ''

        sp = ''
        km = ''
        inicio_dados = 0

        primeiro = tokens[0].upper()
        if primeiro.startswith('SP'):
            sp = tokens[0]
            inicio_dados = 1
            if len(tokens) > 1 and re.search(r'\d+\+\d+', tokens[1]):
                km = tokens[1]
                inicio_dados = 2
        elif re.fullmatch(r'\d+', tokens[0]):
            sp = tokens[0]
            inicio_dados = 1
            if len(tokens) > 2 and tokens[1].upper() == 'KM' and re.search(r'\d+\+\d+', tokens[2]):
                km = f'{tokens[1]} {tokens[2]}'
                inicio_dados = 3
            elif len(tokens) > 1 and re.search(r'\d+\+\d+', tokens[1]):
                km = tokens[1]
                inicio_dados = 2

        return sp, km, self._normalizar_texto_pdf(' '.join(tokens[inicio_dados:]))

    def _validar_passagem(self, passagem: dict) -> tuple[str, list[str]]:
        alertas = []

        # Para Ecovias, os campos de tabela variam bastante. So estes campos
        # bloqueiam a leitura; o restante pode ser bruto/copiado no Excel.
        obrigatorios = [
            'id',
            'data',
            'hora',
            'placa',
        ]
        for campo in obrigatorios:
            if not str(passagem.get(campo, '')).strip():
                alertas.append(f'{campo} vazio')

        if not self.regex_placa.fullmatch(str(passagem.get('placa', '')).upper()):
            alertas.append('placa fora do padrao')

        try:
            datetime.strptime(passagem.get('data', ''), '%d/%m/%Y')
        except ValueError:
            alertas.append('data invalida')

        try:
            datetime.strptime(passagem.get('hora', ''), '%H:%M:%S')
        except ValueError:
            alertas.append('hora invalida')

        status = 'OK' if not alertas else 'Revisar - dados da passagem'
        return status, alertas

    def _extrair_imagens_pdf(self, caminho_pdf: str) -> list[bytes]:
        imagens: list[tuple[int, bytes]] = []

        doc = fitz.open(caminho_pdf)
        try:
            for page in doc:
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    img_data = doc.extract_image(xref)

                    largura = img_data.get('width', 0)
                    altura = img_data.get('height', 0)
                    area = largura * altura

                    if area <= 0:
                        continue

                    imagens.append((area, img_data['image']))
        finally:
            doc.close()

        imagens.sort(key=lambda item: item[0], reverse=True)

        fotos = [img for area, img in imagens if area >= _MIN_AREA_FOTO_PX]
        if fotos:
            return fotos

        # Se o PDF da Ecovias trouxer fotos menores, ainda assim seguimos:
        # o YOLO decide depois se existe placa na imagem.
        return [img for area, img in imagens[:6] if area >= 10_000]

    def processar(self, pasta_pdf: str, caminho_saida: str):
        arquivos_pdf = glob(os.path.join(pasta_pdf, '*.pdf'))
        if not arquivos_pdf:
            print('Nenhum arquivo .pdf encontrado na pasta informada.')
            return

        registros = []

        for caminho_pdf in arquivos_pdf:
            print(f'\nProcessando: {caminho_pdf}')

            print('Extraindo dados da passagem...')
            passagem = self._ler_passagem_pdf(caminho_pdf)
            if passagem is None:
                print('  Pulando - dados nao encontrados.')
                continue

            print('Extraindo fotos embutidas...')
            fotos = self._extrair_imagens_pdf(caminho_pdf)
            print(f'  {len(fotos)} foto(s) encontrada(s).')

            print('Iniciando analise YOLO + OCR...')
            resultado = self._avaliar_bloco(passagem['placa'], fotos)

            registros.append({
                'Arquivo': os.path.basename(caminho_pdf),
                'ID': passagem['id'],
                'Data': passagem['data'],
                'Hora': passagem['hora'],
                'Placa Esperada': passagem['placa'],
                'Eixos': passagem['eixos'],
                'Status Passagem': passagem['status_passagem'],
                'Alertas Passagem': passagem['alertas_passagem'],
                'Placa OCR': resultado['placa_ocr'],
                'OCR Bruto': resultado['ocr_bruto'],
                'Status OCR': resultado['status'],
                'Diferenca OCR': resultado['diferenca'],
                'Imagem Usada': resultado['imagem_usada'],
                'Total Imagens': resultado['total_imgs'],
                'Imagens c/ OCR': resultado['imgs_com_ocr'],
            })

        if not registros:
            print('\nNenhum registro gerado. Verifique os arquivos de entrada.')
            return

        df = pd.DataFrame(registros)
        df.to_excel(caminho_saida, index=False)

        total = len(df)
        aprovados = (df['Status OCR'] == 'Aprovado - busca nas imagens').sum()
        revisar = (df['Status OCR'] == 'Revisar').sum()
        sem_img = (df['Status OCR'] == 'Revisar - sem imagem').sum()
        passagem_ok = (df['Status Passagem'] == 'OK').sum()

        print(f"\n{'='*54}")
        print(f"  Resultado final: {total} passagens")
        print(f"  Passagem OK:     {passagem_ok:3d} ({passagem_ok/total*100:.0f}%)")
        print(f"  Aprovado OCR:    {aprovados:3d} ({aprovados/total*100:.0f}%)")
        print(f"  Revisar OCR:     {revisar:3d} ({revisar/total*100:.0f}%)")
        print(f"  Sem imagem:      {sem_img:3d} ({sem_img/total*100:.0f}%)")
        print(f"{'='*54}")
        print(f"  Arquivo gerado: {caminho_saida}")


if __name__ == '__main__':
    PATH_YOLO = MODEL_PATH

    PASTA_PDFS = DOWNLOADS_DIR / 'evasoes_ecovias'

    PATH_OUT = ensure_parent(RESULTS_EVASAO_DIR / 'resultado_ecovias.xlsx')

    validador = ValidadorPDFEcovias(
        yolo_weights_path=PATH_YOLO,
        usar_gpu=True,
    )

    validador.processar(
        PASTA_PDFS,
        PATH_OUT,
    )




