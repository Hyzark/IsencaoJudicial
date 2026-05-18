"""
ValidadorExcelPro v2
====================
Combina a lÃ³gica de leitura por BLOCO do Excel (app_excel.py) com todo
o pipeline de tratamento de imagem do app.py (YOLO + prÃ©-processamento
multi-filtro + OCR com mÃ¡scara + distÃ¢ncia de Levenshtein customizada).

Estrutura do Excel
------------------
  L006  col_A = int positivo  â†’ INÃCIO do bloco (dados da passagem)
  L007  col_A = '.'           â†’ linhas do bloco  (imagens ancoradas aqui)
  ...
  L049  col_A = '.'           â†’ fim do bloco
  L050  col_A = int positivo  â†’ INÃCIO do prÃ³ximo bloco

Cada bloco contÃ©m 4 imagens (2 frente + 2 traseira).
O sistema processa todas, elege a melhor leitura e gera 1 registro por passagem.

Resultado no Excel
------------------
  Seq | ID | Data | Hora | SP | KM | PraÃ§a | Pista | Sentido | MunicÃ­pio |
  CÃ³digo | Placa Esperada | Modelo | Categoria | Eixos | Tarifa | ResponsÃ¡vel |
  Placa OCR | OCR Bruto | Status OCR | DiferenÃ§a OCR | Imagem Usada |
  Total Imagens | Imagens c/ OCR | Linha Excel
"""

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
import re
import pandas as pd
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
from openpyxl import load_workbook
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from glob import glob
from config import DOWNLOADS_DIR, MODEL_PATH, RESULTS_EVASAO_DIR, ensure_parent
import os

# â”€â”€ Namespaces XML internos do .xlsx â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_NS = {
    'xdr': 'http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing',
    'a':   'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r':   'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'rel': 'http://schemas.openxmlformats.org/package/2006/relationships',
}

_LOGO_FILENAME = 'image1.png'   # logo da empresa â€” ignorado no OCR


class ValidadorExcelPro:

    def __init__(self, yolo_weights_path: str, usar_gpu: bool = True):
        # â”€â”€ Modelos â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.yolo_model = YOLO(yolo_weights_path)
        if usar_gpu:
            self.yolo_model.to('cuda')

        try:
            self.ocr_reader = easyocr.Reader(['pt'], gpu=usar_gpu)
        except Exception:
            self.ocr_reader = easyocr.Reader(['pt'], gpu=True)

        # â”€â”€ ConfiguraÃ§Ãµes (mesmas do app.py) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.ocr_allowlist          = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        self.max_diferencas_aprovacao = 3       # distÃ¢ncia mÃ¡xima para APROVADO
        self.largura_minima_ocr     = 600       # largura mÃ­nima do recorte para OCR

        # Regex de validaÃ§Ã£o de placa (antiga: ABC1234 | Mercosul: ABC1D23)
        self.regex_placa = re.compile(r'([A-Z]{3}\d[A-Z\d]\d{2}|[A-Z]{3}\d{4})')

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # BLOCO A â€” helpers de normalizaÃ§Ã£o / correÃ§Ã£o de placa  (port. de app.py)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    def _limpar_texto_ocr(self, texto: str) -> str:
        """Remove tudo que nÃ£o seja letra ou dÃ­gito e corrige ordem de moto."""
        texto = re.sub(r'[^A-Z0-9]', '', texto.upper())

        # Moto padrÃ£o antigo: OCR lÃª linha de baixo primeiro â†’ 1234ABC â†’ ABC1234
        if re.fullmatch(r'\d{4}[A-Z]{3}', texto):
            texto = texto[4:] + texto[:4]
        # Moto Mercosul: 1D23ABC â†’ ABC1D23
        elif re.fullmatch(r'\d[A-Z]\d{2}[A-Z]{3}', texto):
            texto = texto[4:] + texto[:4]

        return texto

    def _descobrir_mascara(self, placa: str) -> str | None:
        """Detecta se a placa Ã© Antiga (LLLDDDD) ou Mercosul (LLLDLDD)."""
        placa_limpa = re.sub(r'[^A-Z0-9]', '', str(placa).upper())
        if not placa_limpa or len(placa_limpa) != 7:
            return None
        return 'LLLDLDD' if placa_limpa[4].isalpha() else 'LLLDDDD'

    def _corrigir_por_mascara(self, texto: str, mascara: str) -> str:
        """
        Corrige caracteres confundÃ­veis usando a mÃ¡scara L/D da placa.
        L = posiÃ§Ã£o de Letra â†’ converte dÃ­gitos parecidos em letras.
        D = posiÃ§Ã£o de DÃ­gito â†’ converte letras parecidas em dÃ­gitos.
        """
        letras  = {'0':'O','1':'I','2':'Z','4':'A','5':'S','6':'G','8':'B','7':'Z'}
        digitos = {
            'O':'0','Q':'0','D':'0',
            'I':'1','L':'1','T':'1',
            'Z':'2',
            'A':'4',
            'S':'5',
            'G':'6',
            'B':'8','E':'8',
            'J':'1',
            'U':'0',
        }
        corrigido = []
        for char, esperado in zip(texto, mascara):
            if esperado == 'L':
                corrigido.append(letras.get(char, char))
            else:
                corrigido.append(digitos.get(char, char))
        return ''.join(corrigido)

    def _normalizar_placa_ocr(self, texto: str, mascara_esperada: str | None = None) -> str:
        """
        Pipeline completo de normalizaÃ§Ã£o:
        1. Limpa e corrige ordem (moto)
        2. Se jÃ¡ bate no regex â†’ retorna direto
        3. Desliza uma janela de 7 chars tentando corrigir por mÃ¡scara
        4. Retorna o melhor candidato (prefere posiÃ§Ã£o mais Ã  direita)
        """
        texto = self._limpar_texto_ocr(texto)
        if not texto:
            return ''

        if self.regex_placa.fullmatch(texto):
            return texto

        mascaras   = [mascara_esperada] if mascara_esperada else ['LLLDDDD', 'LLLDLDD']
        candidatos = []

        for inicio in range(max(0, len(texto) - 6)):
            trecho = texto[inicio:inicio + 7]
            if len(trecho) != 7:
                continue
            for mascara in mascaras:
                corrigido = self._corrigir_por_mascara(trecho, mascara)
                if self.regex_placa.fullmatch(corrigido):
                    candidatos.append((inicio, corrigido))

        if not candidatos:
            return texto[:7]

        # Prefere o trecho mais Ã  direita (caractere extra costuma aparecer Ã  esquerda)
        candidatos.sort(key=lambda x: -x[0])
        return candidatos[0][1]

    def _distancia_placas(self, placa_texto: str, placa_ocr: str) -> float:
        """
        Levenshtein customizado: pares confundÃ­veis pelo OCR custam 0.35
        em vez de 1.0, evitando rejeiÃ§Ãµes por erros clÃ¡ssicos (Oâ†”0, Iâ†”1â€¦).
        """
        placa_texto = self._limpar_texto_ocr(placa_texto)
        placa_ocr   = self._limpar_texto_ocr(placa_ocr)

        if not placa_texto or not placa_ocr:
            return 999

        confundiveis = {
            ('0','O'),('O','0'),
            ('1','I'),('I','1'),('1','L'),('L','1'),('T','I'),('I','T'),
            ('2','Z'),('Z','2'),('Z','7'),('7','Z'),
            ('4','A'),('A','4'),
            ('5','S'),('S','5'),
            ('6','G'),('G','6'),
            ('8','B'),('B','8'),('B','E'),('E','B'),
            ('D','0'),('0','D'),('D','O'),('O','D'),
        }

        anterior = list(range(len(placa_ocr) + 1))
        for i, char_texto in enumerate(placa_texto, start=1):
            atual = [i]
            for j, char_ocr in enumerate(placa_ocr, start=1):
                if char_texto == char_ocr:
                    custo = 0
                elif (char_texto, char_ocr) in confundiveis:
                    custo = 0.35
                else:
                    custo = 1
                atual.append(min(
                    anterior[j] + 1,
                    atual[j - 1] + 1,
                    anterior[j - 1] + custo,
                ))
            anterior = atual

        return anterior[-1]

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # BLOCO B â€” prÃ©-processamento de imagem  (port. de app.py)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    def _pre_processar_imagem(self, img_array: np.ndarray) -> list[np.ndarray]:

        if img_array is None or img_array.size == 0:
            return []

        _, largura = img_array.shape[:2]

        escala = max(
            3.0,
            self.largura_minima_ocr / max(largura, 1)
        )

        img_array = cv2.resize(
            img_array,
            None,
            fx=escala,
            fy=escala,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(
            img_array,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.fastNlMeansDenoising(
            gray,
            None,
            10,
            7,
            21
        )

        # ============================================
        # CLAHE
        # ============================================

        clahe = cv2.createCLAHE(
            clipLimit=1.2,
            tileGridSize=(8, 8)
        )

        contraste = clahe.apply(gray)

        # ============================================
        # OTSU INV
        # ============================================

        _, otsu_inv = cv2.threshold(
            contraste,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        return [
            img_array,
            contraste,
            otsu_inv
        ]

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # BLOCO C â€” YOLO + OCR numa imagem  (port. de app.py)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    def _processar_imagem_yolo_ocr(self, imagem_bytes: bytes,
                                    mascara_esperada: str | None = None) -> str:
        """
        1. Converte bytes â†’ OpenCV
        2. Roda YOLO (conf â‰¥ 0.35, razÃ£o de proporÃ§Ã£o 0.8â€“6.0)
        3. Para cada detecÃ§Ã£o: gera 5 variantes com _pre_processar_imagem
        4. Roda EasyOCR em cada variante com beamsearch
        5. Pontua candidatos (confianÃ§a + bÃ´nus regex + penalidade tamanho)
        6. Retorna melhor texto ou mensagem de falha
        """
        if not imagem_bytes:
            return 'Nao foi possivel identificar os caracteres'

        pil_img = Image.open(BytesIO(imagem_bytes)).convert('RGB')
        img_cv2 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        resultados_yolo = self.yolo_model(
            img_cv2, imgsz=640, conf=0.25, iou=0.45,
            device=0, verbose=False
        )

        melhor_texto = ''
        melhor_score = -1

        for r in resultados_yolo:
            boxes_ordenadas = sorted(
                r.boxes,
                key=lambda b: float(b.conf[0]) if b.conf is not None else 0,
                reverse=True,
            )

            for box in boxes_ordenadas[:1]:
                conf_det = float(box.conf[0])
                if conf_det < 0.35:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                larg = x2 - x1
                alt  = y2 - y1
                razao = larg / max(alt, 1)

                # Filtro de proporÃ§Ã£o (0.8 aceita placas de moto, ~quadradas)
                if razao < 0.8 or razao > 6.0:
                    continue

                recorte = img_cv2[y1:y2, x1:x2]
                variantes = self._pre_processar_imagem(recorte)

                for variante in variantes:
                    resultados_ocr = self.ocr_reader.readtext(
                        variante,
                        detail=0,
                        allowlist=self.ocr_allowlist,
                        decoder='greedy',
                        text_threshold=0.35,
                    )

                    # Candidatos individuais + texto unido (caso OCR fragmente)
                    candidatos = [(res[1], float(res[2])) for res in resultados_ocr]
                    if len(resultados_ocr) > 1:
                        texto_unido  = ''.join(res[1] for res in resultados_ocr)
                        conf_media   = float(np.mean([res[2] for res in resultados_ocr]))
                        candidatos.append((texto_unido, conf_media))

                    for texto_bruto, confianca in candidatos:
                        texto = self._normalizar_placa_ocr(texto_bruto, mascara_esperada)
                        if not texto:
                            continue

                        placa_valida = self.regex_placa.fullmatch(texto) is not None
                        score = (confianca
                                 + (5.0 if placa_valida else 0.0)
                                 - abs(len(texto) - 7) * 2.0)

                        if score > melhor_score:
                            melhor_score = score
                            melhor_texto = texto

                        # Atalho: leitura perfeita e confiÃ¡vel â†’ encerra
                        if placa_valida:
                            return texto

        return melhor_texto if melhor_texto else 'Nao foi possivel identificar os caracteres'

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # BLOCO D â€” avaliaÃ§Ã£o do bloco completo  (port. de app.py)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    def _avaliar_bloco(self, placa_esperada: str, imagens_bytes: list[bytes]) -> dict:
        """
        Processa TODAS as imagens do bloco, elege a melhor via distÃ¢ncia
        de Levenshtein e decide o status:

          Aprovado - busca nas imagens  â†’ distÃ¢ncia â‰¤ max_diferencas_aprovacao
          Revisar                       â†’ OCR leu algo mas distÃ¢ncia alta
          Revisar - sem imagem          â†’ nenhuma imagem no bloco
        """
        if not imagens_bytes:
            return {
                'placa_ocr':    'Nao foi possivel identificar os caracteres',
                'ocr_bruto':    '',
                'status':       'Revisar - sem imagem',
                'diferenca':    999,
                'imagem_usada': '',
                'total_imgs':   0,
                'imgs_com_ocr': 0,
            }

        mascara_esperada  = self._descobrir_mascara(placa_esperada)
        placa_texto_limpa = self._normalizar_placa_ocr(placa_esperada, mascara_esperada)

        melhor_placa     = ''
        melhor_distancia = 999
        melhor_indice    = None
        melhor_ocr_bruto = ''
        imgs_com_ocr     = 0
        ocr_cache        = {}   # evita reprocessar mesma imagem

        for indice, img_bytes in enumerate(imagens_bytes):
            # Cache por identidade do objeto bytes
            chave = id(img_bytes)
            if chave not in ocr_cache:
                ocr_cache[chave] = self._processar_imagem_yolo_ocr(
                    img_bytes, mascara_esperada
                )
            ocr_bruto = ocr_cache[chave]

            if ocr_bruto == 'Nao foi possivel identificar os caracteres' or not ocr_bruto:
                continue

            imgs_com_ocr += 1
            placa_ocr  = self._normalizar_placa_ocr(ocr_bruto, mascara_esperada)
            distancia  = self._distancia_placas(placa_texto_limpa, placa_ocr)

            if distancia < melhor_distancia:
                melhor_distancia = distancia
                melhor_placa     = placa_ocr
                melhor_indice    = indice
                melhor_ocr_bruto = ocr_bruto

            # Leitura perfeita â†’ para imediatamente
            if distancia == 0:
                break

        total_imgs = len(imagens_bytes)

        if melhor_distancia <= self.max_diferencas_aprovacao:
            status     = 'Aprovado - busca nas imagens'
            placa_final = placa_texto_limpa
        else:
            status     = 'Revisar'
            placa_final = melhor_placa or 'Nao foi possivel identificar os caracteres'

        return {
            'placa_ocr':    placa_final,
            'ocr_bruto':    melhor_ocr_bruto,
            'status':       status,
            'diferenca':    round(melhor_distancia, 2),
            'imagem_usada': (melhor_indice + 1) if melhor_indice is not None else '',
            'total_imgs':   total_imgs,
            'imgs_com_ocr': imgs_com_ocr,
        }

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # BLOCO E â€” leitura dos blocos do Excel
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    def _ler_blocos(self, ws) -> list[dict]:
        """
        Detecta inÃ­cio de cada passagem pela coluna A ser um inteiro positivo.
        Extrai todos os campos e calcula o intervalo de linhas do bloco.
        """
        blocos = []

        for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
            col_a = row[0]
            if not (isinstance(col_a, int) and col_a > 0):
                continue

            blocos.append({
                'seq':         col_a,
                'linha':       row_idx,
                'id':          str(row[1])  if row[1]  is not None else '',
                'data':        row[2],
                'hora':        row[3],
                'sp':          row[4],
                'km':          str(row[5])  if row[5]  is not None else '',
                'praca':       str(row[6])  if row[6]  is not None else '',
                'pista':       row[7],
                'sentido':     str(row[8])  if row[8]  is not None else '',
                'municipio':   str(row[9])  if row[9]  is not None else '',
                'codigo':      row[10],
                'placa':       str(row[11]) if row[11] is not None else '',
                'modelo':      str(row[12]) if row[12] is not None else '',
                'categoria':   str(row[13]) if row[13] is not None else '',
                'eixos':       row[14],
                'tarifa':      row[15],
                'responsavel': str(row[16]) if row[16] is not None else '',
                'linha_fim':   None,
            })

        for i, b in enumerate(blocos):
            b['linha_fim'] = blocos[i + 1]['linha'] if i + 1 < len(blocos) else 99999

        return blocos

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # BLOCO F â€” mapeamento de imagens via XML interno do .xlsx
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    def _mapear_imagens_xml(self, caminho_excel: str) -> dict[int, list[bytes]]:
        """
        LÃª os XMLs internos do .xlsx (que Ã© um ZIP) para montar:
            { linha_ancora_inicio: [bytes_img, ...] }

        Motivo: o openpyxl perde a referÃªncia ao rId que liga cada Ã¢ncora
        ao arquivo de mÃ­dia correto, fazendo todos os blocos receberem a
        mesma imagem. Aqui lemos drawing.xml + drawing.xml.rels diretamente.
        """
        mapa: dict[int, list[bytes]] = {}

        with zipfile.ZipFile(caminho_excel, 'r') as z:
            arquivos = z.namelist()

            # Descobre o drawing da sheet1 via seu .rels
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

            # .rels do drawing: rId â†’ nome_arquivo
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

            # PrÃ©-carrega todos os bytes de mÃ­dia
            bytes_midia = {
                arq.split('/')[-1]: z.read(arq)
                for arq in arquivos
                if arq.startswith('xl/media/')
            }

            # drawing XML: Ã¢ncora â†’ rId â†’ arquivo â†’ bytes
            drawing_root = ET.fromstring(z.read(drawing_path).decode())
            for anchor in drawing_root.findall('xdr:twoCellAnchor', _NS):
                from_elem = anchor.find('xdr:from', _NS)
                if from_elem is None:
                    continue
                # drawing usa Ã­ndice 0-based; +1 converte para linha Excel
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

                if not nome_arquivo or nome_arquivo == _LOGO_FILENAME:
                    continue

                img_bytes = bytes_midia.get(nome_arquivo)
                if img_bytes:
                    mapa.setdefault(row_inicio, []).append(img_bytes)

        return mapa

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # BLOCO G â€” pipeline principal
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    def processar(self, pasta_excel: str, caminho_saida: str):

        arquivos_excel = glob(os.path.join(pasta_excel, '*.xlsx'))
        if not arquivos_excel:
            print('Nenhum arquivo .xlsx encontrado na pasta informada.')
            return

        registros = []

        for caminho_excel in arquivos_excel:

            print(f'\nProcessando: {caminho_excel}')

            print('Carregando planilha...')
            wb = load_workbook(caminho_excel, data_only=True)
            ws = wb.active

            print('Lendo blocos de passagem...')
            blocos = self._ler_blocos(ws)

            print(f'  {len(blocos)} passagens encontradas.')

            print('Mapeando imagens via XML interno...')
            mapa_bytes = self._mapear_imagens_xml(caminho_excel)

            # Associa bytes de imagem a cada bloco pelo intervalo de linhas
            for b in blocos:

                imgs = []

                for linha_ancora, lista_bytes in mapa_bytes.items():

                    if b['linha'] <= linha_ancora < b['linha_fim']:
                        imgs.extend(lista_bytes)

                b['_imagens_bytes'] = imgs

            print('Iniciando anÃ¡lise YOLO + OCR por bloco...')

            for b in tqdm(blocos):

                resultado = self._avaliar_bloco(
                    b['placa'],
                    b['_imagens_bytes']
                )

                registros.append({

                    'Arquivo': os.path.basename(caminho_excel),

                    'Seq': b['seq'],
                    'ID Sistema': b['id'],
                    'Data': b['data'],
                    'Placa Esperada': b['placa'],
                    'Categoria': b['categoria'],
                    'Eixos': b['eixos'],
                    'Placa OCR': resultado['placa_ocr'],
                    'OCR Bruto': resultado['ocr_bruto'],
                    'Status OCR': resultado['status'],
                    'DiferenÃ§a OCR': resultado['diferenca'],
                    'Imagem Usada': resultado['imagem_usada'],
                    'Total Imagens': resultado['total_imgs'],
                    'Imagens c/ OCR': resultado['imgs_com_ocr'],
                    'Linha Excel': b['linha'],
                })

        if not registros:
            print('Nenhum registro gerado. Verifique os arquivos de entrada.')
            return

        df = pd.DataFrame(registros)

        df.to_excel(caminho_saida, index=False)

        # â”€â”€ Resumo â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        total = len(df)

        aprovados = (
            df['Status OCR'] == 'Aprovado - busca nas imagens'
        ).sum()

        revisar = (
            df['Status OCR'] == 'Revisar'
        ).sum()

        sem_img = (
            df['Status OCR'] == 'Revisar - sem imagem'
        ).sum()

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

    PASTA_EXCELS = DOWNLOADS_DIR / 'evasoes_spvias'

    PATH_OUT = ensure_parent(RESULTS_EVASAO_DIR / 'resultado_spvias.xlsx')

    validador = ValidadorExcelPro(
        yolo_weights_path=PATH_YOLO,
        usar_gpu=True
    )

    validador.processar(
        PASTA_EXCELS,
        PATH_OUT
    )





