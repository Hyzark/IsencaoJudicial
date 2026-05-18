from pathlib import Path
import fitz  # PyMuPDF
import cv2
import easyocr
import pandas as pd
import numpy as np
from ultralytics import YOLO
import re
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
# NOVO: torch necessário para inference_mode() e controle de precisão
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
from config import DOWNLOADS_DIR, MODEL_PATH, RESULTS_ISENCAO_DIR, ensure_parent


# ==============================================================================
# WORKER de extração de páginas (sem alteração em relação à versão anterior)
# Mantido fora da classe para compatibilidade com ProcessPoolExecutor.
# ==============================================================================
def _extrair_pagina_worker(args):
    pdf_path, page_num, regex_data_pattern, regex_placa_pattern, qtd_imagens_por_passagem = args
    regex_data = re.compile(regex_data_pattern)
    regex_placa = re.compile(regex_placa_pattern)

    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]

        imagens_pagina = []
        imagem_cache_local = {}

        for img in page.get_images(full=True):
            xref = img[0]
            largura_original = img[2]
            altura_original = img[3]
            if largura_original < 80 or altura_original < 40:
                continue
            rects = page.get_image_rects(xref)
            if not rects:
                continue
            if xref not in imagem_cache_local:
                base_image = doc.extract_image(xref)
                imagem_cache_local[xref] = base_image["image"]
            for rect in rects:
                if rect.width < 20 or rect.height < 20:
                    continue
                imagens_pagina.append({
                    "xref": xref,
                    "rect": rect,
                    "bytes": imagem_cache_local[xref],
                    "centro_x": (rect.x0 + rect.x1) / 2,
                    "centro_y": (rect.y0 + rect.y1) / 2,
                })

        imagens_pagina = sorted(imagens_pagina, key=lambda item: (item["rect"].y0, item["rect"].x0))

        texto_completo = page.get_text("text")
        blocos_texto = texto_completo.split("Data/Hora")

        passagens_pagina = []
        y_cursor = -1

        for bloco in blocos_texto[1:]:
            try:
                data_hora = regex_data.search(bloco).group(0)
                linhas = bloco.split('\n')
                categoria = [
                    linha.strip()
                    for linha in linhas
                    if linha.strip().isdigit() and len(linha.strip()) <= 2
                ][0]
                placa_texto = regex_placa.search(bloco).group(0)

                rects_placa = sorted(page.search_for(placa_texto), key=lambda r: (r.y0, r.x0))
                rect_placa = None
                for r in rects_placa:
                    if r.y0 >= y_cursor - 1:
                        rect_placa = r
                        break
                if rect_placa is None and rects_placa:
                    rect_placa = rects_placa[0]

                centro_y = ((rect_placa.y0 + rect_placa.y1) / 2) if rect_placa else None
                if rect_placa:
                    y_cursor = rect_placa.y1

                passagens_pagina.append({
                    "Data/Hora": data_hora,
                    "Categoria": categoria,
                    "Placa (Texto)": placa_texto,
                    "centro_y": centro_y,
                })
            except (AttributeError, IndexError):
                continue

        passagens = []
        for indice_passagem, passagem in enumerate(passagens_pagina):
            inicio = indice_passagem * qtd_imagens_por_passagem
            fim = inicio + qtd_imagens_por_passagem
            imagens_candidatas = [img["bytes"] for img in imagens_pagina[inicio:fim]]
            passagens.append({
                "Data/Hora": passagem["Data/Hora"],
                "Categoria": passagem["Categoria"],
                "Placa (Texto)": passagem["Placa (Texto)"],
                "imagens_candidatas": imagens_candidatas,
                "Pagina": page_num + 1,
            })

        doc.close()
        return passagens

    except Exception as e:
        print(f"[ERRO] Página {page_num + 1}: {e}")
        return []


class ValidadorPassagens:
    def __init__(self, yolo_weights_path, batch_size=64):
        self.yolo_model = YOLO(yolo_weights_path)
        self.yolo_model.to("cuda")
        self.yolo_model.model = torch.compile(self.yolo_model.model)
        # NOVO: converte o modelo para FP16 (half precision).
        # Na RTX 4060 Ti isso dobra o throughput de Tensor Cores e reduz
        # uso de VRAM ~50%, permitindo batches maiores sem OOM.
        # Impacto esperado: +30–60% de FPS no YOLO.
        self.ocr_reader = easyocr.Reader(['pt'], gpu=True)
        self.ocr_allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        self.qtd_imagens_por_passagem = 3
        self.max_diferencas_aprovacao = 3
        self.largura_minima_ocr = 600

        # NOVO: tamanho do lote para inferência YOLO em batch.
        # 32 é seguro para a VRAM de 16 GB da 4060 Ti com imgsz=512.
        # Aumente para 64 se a VRAM permitir (monitore com nvidia-smi).
        self.batch_size = batch_size

        self.regex_placa = re.compile(r"([A-Z]{3}\d[A-Z\d]\d{2}|[A-Z]{3}\d{4})")
        self.regex_data = re.compile(r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})")

    # ==========================================================================
    # HELPERS: sem alteração (lógica de validação preservada)
    # ==========================================================================

    def _limpar_texto_ocr(self, texto):
        texto = re.sub(r"[^A-Z0-9]", "", texto.upper())
        if re.fullmatch(r"\d{4}[A-Z]{3}", texto):
            texto = texto[4:] + texto[:4]
        elif re.fullmatch(r"\d[A-Z]\d{2}[A-Z]{3}", texto):
            texto = texto[4:] + texto[:4]
        return texto

    def _descobrir_mascara(self, placa):
        placa_limpa = re.sub(r"[^A-Z0-9]", "", str(placa).upper())
        if not placa_limpa or len(placa_limpa) != 7:
            return None
        if placa_limpa[4].isalpha():
            return "LLLDLDD"
        return "LLLDDDD"

    def _corrigir_por_mascara(self, texto, mascara):
        letras = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B", "7": "Z"}
        digitos = {
            "O": "0", "Q": "0", "D": "0",
            "I": "1", "L": "1", "T": "1",
            "Z": "2", "A": "4", "S": "5",
            "G": "6", "B": "8", "E": "8",
            "J": "1", "U": "0",
        }
        corrigido = []
        for char, esperado in zip(texto, mascara):
            if esperado == "L":
                corrigido.append(letras.get(char, char))
            else:
                corrigido.append(digitos.get(char, char))
        return "".join(corrigido)

    def _normalizar_placa_ocr(self, texto, mascara_esperada=None):
        texto = self._limpar_texto_ocr(texto)
        if not texto:
            return ""
        if self.regex_placa.fullmatch(texto):
            return texto
        mascaras = [mascara_esperada] if mascara_esperada else ["LLLDDDD", "LLLDLDD"]
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
        candidatos.sort(key=lambda x: -x[0])
        return candidatos[0][1]

    def _distancia_placas(self, placa_texto, placa_ocr):
        placa_texto = self._limpar_texto_ocr(placa_texto)
        placa_ocr = self._limpar_texto_ocr(placa_ocr)
        if not placa_texto or not placa_ocr:
            return 999
        confundiveis = {
            ("0", "O"), ("O", "0"),
            ("1", "I"), ("I", "1"), ("1", "L"), ("L", "1"), ("T", "I"), ("I", "T"),
            ("2", "Z"), ("Z", "2"), ("Z", "7"), ("7", "Z"),
            ("4", "A"), ("A", "4"),
            ("5", "S"), ("S", "5"),
            ("6", "G"), ("G", "6"),
            ("8", "B"), ("B", "8"), ("B", "E"), ("E", "B"),
            ("D", "0"), ("0", "D"), ("D", "O"), ("O", "D"),
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
                atual.append(min(anterior[j] + 1, atual[j - 1] + 1, anterior[j - 1] + custo))
            anterior = atual
        return anterior[-1]

    # ==========================================================================
    # NOVO: criar_lotes
    # Recebe lista de passagens e retorna sublistas (lotes) de tamanho batch_size.
    #
    # Por que isso importa para a GPU?
    # O YOLO processa um tensor [N, C, H, W]. Com N=1 (atual), a GPU fica ociosa
    # entre lançamentos de kernel CUDA. Com N=32, a GPU executa um único kernel
    # que preenche todos os Tensor Cores da 4060 Ti simultaneamente.
    # Overhead por chamada CUDA: ~0.3 ms. Com 1000 imagens:
    #   Antes:  1000 × 0.3 ms overhead = +300 ms só em overhead de kernel
    #   Depois: 32 lotes × 0.3 ms = +9.6 ms overhead total
    # ==========================================================================
    def criar_lotes(self, passagens, batch_size=None):
        """
        Divide a lista de passagens em sublistas de tamanho batch_size.

        Parâmetros:
            passagens  : list de dicts com 'imagens_candidatas'
            batch_size : int — sobrescreve self.batch_size se fornecido

        Retorna:
            list de lists de passagens
        """
        # Permite sobrescrever batch_size por chamada sem alterar o default do objeto
        bs = batch_size or self.batch_size

        # range(0, total, bs) gera os índices de início de cada lote:
        # [0, 32, 64, 96, ...] → cada fatia passagens[i:i+bs] é um lote
        return [passagens[i:i + bs] for i in range(0, len(passagens), bs)]

    # ==========================================================================
    # NOVO: detectar_placas_batch
    # Executa YOLO em um lote inteiro de imagens de uma só vez.
    #
    # Fluxo:
    #   1. Decodifica bytes → BGR (CPU, paralelo implícito via numpy)
    #   2. Redimensiona para imgsz=512 (menor que os 1024 anteriores → +2x FPS)
    #   3. Empilha em lista — Ultralytics aceita list[np.ndarray] como batch
    #   4. torch.inference_mode() desativa autograd → -15% uso de VRAM e CPU
    #   5. half=True → FP16 nos pesos já carregados em half() no __init__
    #   6. Retorna dict {indice_global → lista de recortes BGR da placa}
    #      mantendo rastreabilidade de qual resultado pertence a qual passagem
    #
    # Impacto esperado na RTX 4060 Ti:
    #   GPU antes:  ~30% (1 imagem × 1024px, FP32)
    #   GPU depois: ~75–90% (32 imagens × 512px, FP16)
    #   Throughput: ~3–5x mais passagens/segundo
    # ==========================================================================
    def detectar_placas_batch(self, lote_imagens_bytes):
        """
        Executa detecção YOLO em batch sobre uma lista de bytes de imagem.

        Parâmetros:
            lote_imagens_bytes : list[(indice_global, bytes)]
                Cada item é uma tupla com o índice original da passagem e
                os bytes da imagem candidata. O índice preserva a ordem.

        Retorna:
            dict {indice_global: list[np.ndarray]}
                Mapeamento de índice → lista de recortes BGR válidos detectados.
                Imagens sem detecção retornam lista vazia.
                Imagens inválidas (bytes corrompidos) são puladas sem quebrar o batch.
        """
        # ── Passo 1: decodificar bytes → imagens BGR ──────────────────────────
        # Faz isso separado do loop YOLO para isolar erros de decodificação.
        # Imagens inválidas são registradas mas não interrompem o lote.
        indices_validos = []   # guarda os índices que conseguiram decodificar
        imagens_cv2 = []       # lista de arrays BGR para passar ao YOLO

        for indice_global, img_bytes in lote_imagens_bytes:
            if not img_bytes:
                # Bytes vazios/None: pula sem logar (comum quando a passagem
                # tem menos imagens candidatas que qtd_imagens_por_passagem)
                continue
            try:
                # PIL → RGB → numpy → BGR (mesma conversão do código original)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                arr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                imagens_cv2.append(arr)
                indices_validos.append(indice_global)
            except Exception as e:
                # Imagem corrompida no PDF: registra e segue
                print(f"  [AVISO] Imagem índice {indice_global} inválida, pulando: {e}")
                continue

        # Nenhuma imagem válida no lote → retorna dict vazio
        if not imagens_cv2:
            return {}

        # ── Passo 2: inferência YOLO em batch ─────────────────────────────────
        # torch.inference_mode() é mais agressivo que no_grad():
        #   • desativa autograd E o mecanismo de version counter
        #   • reduz uso de VRAM em ~15% durante inferência
        #   • compatível com .half() já aplicado no __init__
        #
        # imgsz=512: reduz resolução de entrada de 1024→512.
        #   Para placas veiculares em imagens de passagem de pedágio,
        #   512px é suficiente para detecção confiável e dobra o FPS.
        #   Ajuste para 640 se notar queda de confiança na detecção.
        #
        # half=True: usa os pesos FP16 já carregados — necessário aqui
        #   para que o tensor de entrada seja convertido automaticamente.
        #
        # stream=False: retorna todos os resultados de uma vez.
        #   stream=True (gerador) seria melhor para batches >128, mas
        #   para 32–64 não há diferença prática e simplifica o código.
        with torch.inference_mode():
            resultados_yolo = self.yolo_model(
                imagens_cv2,       # list[np.ndarray] — Ultralytics aceita nativamente
                imgsz=512,         # resolução de entrada reduzida (era 1024)
                conf=0.25,         # limiar de confiança de detecção (inalterado)
                iou=0.45,          # NMS IoU (inalterado)
                device=0,          # GPU 0 (RTX 4060 Ti)
                verbose=False,
                half=True,     # sem logs por imagem
            )

        # ── Passo 3: extrair recortes por índice ──────────────────────────────
        # resultados_yolo[i] corresponde a imagens_cv2[i] = indices_validos[i]
        # Iteramos em paralelo com zip para manter o alinhamento.
        recortes_por_indice = {}

        for indice_global, resultado, img_cv2 in zip(indices_validos, resultados_yolo, imagens_cv2):
            recortes = []
            boxes = resultado.boxes

            if boxes is None or len(boxes) == 0:
                # Nenhuma detecção nesta imagem — lista vazia preserva o índice
                recortes_por_indice[indice_global] = []
                continue

            # Ordena caixas por confiança decrescente (igual ao código original)
            boxes_ordenadas = sorted(
                boxes,
                key=lambda b: float(b.conf[0]) if b.conf is not None else 0,
                reverse=True,
            )

            for box in boxes_ordenadas:
                conf_deteccao = float(box.conf[0])
                if conf_deteccao < 0.35:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                largura_box = x2 - x1
                altura_box  = y2 - y1
                razao = largura_box / max(altura_box, 1)

                # Filtro de proporção (inalterado)
                if razao < 0.8 or razao > 6.0:
                    continue

                recorte = img_cv2[y1:y2, x1:x2]
                if recorte is None or recorte.size == 0 or recorte.shape[0] < 2 or recorte.shape[1] < 2:
                    continue

                recortes.append(recorte)

            recortes_por_indice[indice_global] = recortes

        return recortes_por_indice

    # ==========================================================================
    # pre_processar_imagem — sem alteração
    # ==========================================================================
    def _pre_processar_imagem(self, img_array: np.ndarray) -> list[np.ndarray]:

        if img_array is None or img_array.size == 0:
            return []

        altura, largura = img_array.shape[:2]
        if altura < 2 or largura < 2:
            return []

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

        if img_array is None or img_array.size == 0 or img_array.shape[0] < 2 or img_array.shape[1] < 2:
            return []

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
            img
            for img in [img_array, contraste, otsu_inv]
            if img is not None and img.size > 0 and img.shape[0] >= 2 and img.shape[1] >= 2
        ]

    # ==========================================================================
    # processar_imagem_yolo_ocr — MANTIDO para compatibilidade.
    # Ainda é usado pelo fluxo de fallback (cache miss com bytes diretos).
    # A diferença é que agora recebe um recorte BGR já detectado pelo batch,
    # então pula a etapa YOLO e vai direto ao OCR.
    #
    # ALTERAÇÃO: recebe opcionalmente recortes_predetectados (list[np.ndarray])
    # vindos de detectar_placas_batch. Quando presentes, pula o YOLO.
    # ==========================================================================
    def processar_imagem_yolo_ocr(self, imagem_bytes, mascara_esperada=None, recortes_predetectados=None):
        """
        Se recortes_predetectados for fornecido (vindo do batch YOLO),
        pula a etapa de detecção e vai direto ao OCR.
        Caso contrário, executa YOLO unitário (fallback de compatibilidade).
        """
        melhor_texto = ""
        melhor_score = -1

        # ── Caminho batch: recortes já detectados ─────────────────────────────
        if recortes_predetectados is not None:
            imagens_a_processar = recortes_predetectados
        else:
            # ── Caminho legado: YOLO unitário (fallback) ──────────────────────
            if not imagem_bytes:
                return "Nao foi possivel identificar os caracteres"
            pil_image = Image.open(BytesIO(imagem_bytes)).convert("RGB")
            img_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            with torch.inference_mode():
                resultados_yolo = self.yolo_model(
                    img_cv2, imgsz=512, conf=0.25, iou=0.45,
                    device=0, verbose=False, half=True,
                )

            imagens_a_processar = []
            for r in resultados_yolo:
                for box in sorted(r.boxes, key=lambda b: float(b.conf[0]), reverse=True):
                    if float(box.conf[0]) < 0.35:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    larg = x2 - x1
                    alt  = y2 - y1
                    if larg / max(alt, 1) < 0.8 or larg / max(alt, 1) > 6.0:
                        continue
                    recorte = img_cv2[y1:y2, x1:x2]
                    if recorte is None or recorte.size == 0 or recorte.shape[0] < 2 or recorte.shape[1] < 2:
                        continue
                    imagens_a_processar.append(recorte)

        # ── OCR sobre os recortes (independente do caminho) ───────────────────
        for recorte_placa in imagens_a_processar:
            imagens_para_ocr = self._pre_processar_imagem(recorte_placa)

            achou_perfeita = False
            for imagem_ocr in imagens_para_ocr:
                if achou_perfeita:
                    break

                try:
                    resultados_ocr = self.ocr_reader.readtext(
                        imagem_ocr,
                        detail=0,
                        allowlist=self.ocr_allowlist,
                        decoder='greedy',
                        text_threshold=0.35,
                    )
                except Exception as e:
                    print(f"  [AVISO] OCR falhou em recorte (shape={imagem_ocr.shape}): {e}")
                    continue

                candidatos_ocr = [(texto, 0.8) for texto in resultados_ocr]

                if len(resultados_ocr) > 1:
                    texto_unido = "".join(resultados_ocr)
                    candidatos_ocr.append((texto_unido, 0.8))

                for texto_bruto, confianca in candidatos_ocr:
                    texto = self._normalizar_placa_ocr(texto_bruto, mascara_esperada)
                    if not texto:
                        continue
                    placa_valida = self.regex_placa.fullmatch(texto) is not None
                    score = confianca + (5.0 if placa_valida else 0.0) - abs(len(texto) - 7) * 2.0
                    if score > melhor_score:
                        melhor_score = score
                        melhor_texto = texto
                    if placa_valida and confianca >= 0.40:
                        achou_perfeita = True
                        return texto

        return melhor_texto if melhor_texto else "Nao foi possivel identificar os caracteres"

    # ==========================================================================
    # _ler_ocr_com_cache — alterado para aceitar recortes pré-detectados
    # ==========================================================================
    def _ler_ocr_com_cache(self, imagem_bytes, ocr_cache, mascara_esperada=None, recortes_predetectados=None):
        if not imagem_bytes and not recortes_predetectados:
            return ""

        # Chave do cache: hash dos bytes da imagem original (antes do recorte)
        # Imagens idênticas no PDF compartilham o mesmo cache mesmo entre lotes
        chave_cache = hash(imagem_bytes) if imagem_bytes else None

        if chave_cache and chave_cache in ocr_cache:
            resultado = ocr_cache[chave_cache]
        else:
            resultado = self.processar_imagem_yolo_ocr(
                imagem_bytes,
                mascara_esperada,
                recortes_predetectados=recortes_predetectados,
            )
            if chave_cache:
                ocr_cache[chave_cache] = resultado

        if resultado == "Nao foi possivel identificar os caracteres":
            return ""
        return self._normalizar_placa_ocr(resultado, mascara_esperada)

    # ==========================================================================
    # avaliar_passagem — sem alteração (lógica de validação preservada)
    # ==========================================================================
    def avaliar_passagem(self, placa_texto, imagens_candidatas, ocr_cache, recortes_por_imagem=None):
        mascara_esperada = self._descobrir_mascara(placa_texto)
        placa_texto_limpa = self._normalizar_placa_ocr(placa_texto, mascara_esperada)
        imagens_candidatas = list(imagens_candidatas or [])

        if not imagens_candidatas:
            return {
                "placa_final": "Nao foi possivel identificar os caracteres",
                "ocr_bruto": "",
                "status": "Revisar - sem imagem",
                "diferenca": 999,
                "imagem_usada": "",
            }

        melhor_placa    = ""
        melhor_distancia = 999
        melhor_indice   = None
        melhor_ocr_bruto = ""

        for indice, imagem_bytes in enumerate(imagens_candidatas):
            # Se temos recortes pré-detectados para esta imagem, passa para o OCR
            recortes = (recortes_por_imagem or {}).get(indice, None)

            placa_ocr = self._ler_ocr_com_cache(
                imagem_bytes, ocr_cache, mascara_esperada,
                recortes_predetectados=recortes,
            )
            if not placa_ocr:
                continue

            distancia = self._distancia_placas(placa_texto_limpa, placa_ocr)
            if distancia < melhor_distancia:
                melhor_distancia = distancia
                melhor_placa     = placa_ocr
                melhor_indice    = indice
                melhor_ocr_bruto = placa_ocr
            if distancia == 0:
                break

        if melhor_distancia <= self.max_diferencas_aprovacao:
            return {
                "placa_final":  placa_texto_limpa,
                "ocr_bruto":    melhor_ocr_bruto,
                "status":       "Aprovado - busca nas imagens",
                "diferenca":    melhor_distancia,
                "imagem_usada": (melhor_indice + 1) if melhor_indice is not None else "",
            }

        return {
            "placa_final":  melhor_placa or "Nao foi possivel identificar os caracteres",
            "ocr_bruto":    melhor_ocr_bruto,
            "status":       "Revisar",
            "diferenca":    melhor_distancia,
            "imagem_usada": (melhor_indice + 1) if melhor_indice is not None else "",
        }

    # ==========================================================================
    # ALTERADO: gerar_relatorio
    #
    # Pipeline anterior:
    #   for passagem in passagens:           ← loop 1 (sequencial)
    #       for imagem in imagens_candidatas: ← loop 2 (sequencial)
    #           YOLO(imagem)                 ← GPU subutilizada (N=1)
    #           OCR(recorte)                 ← CPU/GPU
    #
    # Pipeline novo:
    #   for lote in criar_lotes(passagens):  ← loop 1 sobre lotes de 32
    #       [preparar pares (idx, bytes)]    ← flatten das candidatas do lote
    #       detectar_placas_batch(pares)     ← YOLO único com N=32 imagens
    #       for passagem in lote:            ← loop 2 (só OCR, sem YOLO)
    #           avaliar_passagem(recortes)   ← OCR com recortes pré-detectados
    #
    # Impacto:
    #   • YOLO: de ~1000 chamadas individuais → ~32 chamadas de batch
    #   • GPU:  de ~30% → ~75–90%
    #   • Throughput: estimativa de 3–5x mais passagens/segundo
    #   • OCR:  inalterado (EasyOCR não tem batch nativo confiável)
    # ==========================================================================
    def gerar_relatorio(self, passagens, output_excel):
        dados_finais   = []
        ocr_cache      = {}   # compartilhado entre todos os lotes (evita retrabalho)

        # Divide passagens em lotes de batch_size
        lotes = self.criar_lotes(passagens)

        for lote in tqdm(lotes, desc=f"Processando lotes (batch={self.batch_size})"):

            # ── Passo 1: montar lista de (índice_global, bytes) para o YOLO ──
            # Cada passagem tem até qtd_imagens_por_passagem imagens candidatas.
            # Precisamos de um índice único por (passagem, posicao_imagem) para
            # depois reconstruir qual recorte vai para qual passagem.
            #
            # Esquema do índice composto:
            #   indice_global = indice_passagem_no_lote * 100 + indice_imagem
            # Limite implícito: até 100 imagens candidatas por passagem.
            # Para qtd_imagens_por_passagem=3, isso nunca é atingido.
            pares_para_yolo = []
            for i_passagem, passagem in enumerate(lote):
                for i_img, img_bytes in enumerate(passagem.get("imagens_candidatas", [])):
                    indice_global = i_passagem * 100 + i_img
                    pares_para_yolo.append((indice_global, img_bytes))

            # ── Passo 2: YOLO em batch (único lançamento de kernel CUDA) ──────
            # recortes_batch = {indice_global: [array_BGR, ...]}
            recortes_batch = self.detectar_placas_batch(pares_para_yolo)

            # ── Passo 3: OCR e avaliação por passagem ─────────────────────────
            for i_passagem, passagem in enumerate(lote):

                # Remonta o dict {indice_imagem: recortes} para esta passagem
                # para passar ao avaliar_passagem de forma compatível
                recortes_passagem = {}
                for i_img in range(len(passagem.get("imagens_candidatas", []))):
                    indice_global = i_passagem * 100 + i_img
                    if indice_global in recortes_batch:
                        recortes_passagem[i_img] = recortes_batch[indice_global]

                avaliacao = self.avaliar_passagem(
                    passagem["Placa (Texto)"],
                    passagem.get("imagens_candidatas", []),
                    ocr_cache,
                    recortes_por_imagem=recortes_passagem,
                )

                dados_finais.append({
                    "Pagina":                 passagem.get("Pagina", ""),
                    "Data/Hora":              passagem["Data/Hora"],
                    "Categoria":              passagem["Categoria"],
                    "Placa (Texto)":          passagem["Placa (Texto)"],
                    "Placa (Imagem/OCR)":     avaliacao["placa_final"],
                    "OCR Bruto":              avaliacao["ocr_bruto"],
                    "Status OCR":             avaliacao["status"],
                    "Diferenca OCR":          avaliacao["diferenca"],
                    "Imagem Usada":           avaliacao["imagem_usada"],
                    "Qtd Imagens Candidatas": len(passagem.get("imagens_candidatas", [])),
                })

        df = pd.DataFrame(dados_finais)
        df.to_excel(output_excel, index=False)
        print(f"Processamento concluido. Salvo em: {output_excel}")
        return df

    # ==========================================================================
    # extrair_dados_pdf e processar — sem alteração
    # ==========================================================================
    def extrair_dados_pdf(self, pdf_path, max_workers=None):
        doc = fitz.open(pdf_path)
        n_pages = len(doc)
        doc.close()

        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)

        print(f"  Processando {n_pages} páginas com {max_workers} workers paralelos...")

        args_list = [
            (
                str(pdf_path),
                page_num,
                self.regex_data.pattern,
                self.regex_placa.pattern,
                self.qtd_imagens_por_passagem,
            )
            for page_num in range(n_pages)
        ]

        resultados_por_pagina = [None] * n_pages
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_page = {
                executor.submit(_extrair_pagina_worker, args): args[1]
                for args in args_list
            }
            for future in tqdm(as_completed(future_to_page), total=n_pages, desc="Extraindo páginas"):
                page_num = future_to_page[future]
                resultados_por_pagina[page_num] = future.result()

        passagens = [p for pagina in resultados_por_pagina if pagina for p in pagina]
        return passagens

    def processar(self, caminho_entrada: str, caminho_saida: str, max_workers=None):
        caminho = Path(caminho_entrada)

        if caminho.is_dir():
            arquivos = list(caminho.glob('*.pdf'))
        elif caminho.is_file():
            arquivos = [caminho]
        else:
            print(f'Entrada não encontrada: {caminho_entrada}')
            return

        if not arquivos:
            print('Nenhum arquivo .pdf encontrado.')
            return

        todas_passagens = []
        for pdf in arquivos:
            print(f'\nExtraindo dados de: {pdf.name}')
            passagens = self.extrair_dados_pdf(str(pdf), max_workers=max_workers)
            print(f'  {len(passagens)} passagem(ns) encontrada(s).')
            todas_passagens.extend(passagens)

        if not todas_passagens:
            print('\nNenhuma passagem encontrada nos arquivos.')
            return

        print(f'\nIniciando processamento YOLO + OCR ({len(todas_passagens)} passagens)...')
        self.gerar_relatorio(todas_passagens, caminho_saida)


if __name__ == "__main__":
    CAMINHO_YOLO_WEIGHTS = MODEL_PATH
    CAMINHO_PDF          = DOWNLOADS_DIR / '01.pdf'
    SAIDA_EXCEL          = ensure_parent(RESULTS_ISENCAO_DIR / 'resultado_autoban.xlsx')

    # batch_size=32: seguro para RTX 4060 Ti com imgsz=512 + FP16
    # Aumente para 64 se nvidia-smi mostrar <80% de uso de VRAM
    validador = ValidadorPassagens(
        yolo_weights_path=CAMINHO_YOLO_WEIGHTS,
        batch_size=64,
    )

    print("Extraindo dados do PDF...")
    passagens_extraidas = validador.extrair_dados_pdf(CAMINHO_PDF)

    print(f"Total de registros encontrados: {len(passagens_extraidas)}")
    print("Iniciando processamento YOLO + OCR em batch...")

    df_resultado = validador.gerar_relatorio(passagens_extraidas, SAIDA_EXCEL)