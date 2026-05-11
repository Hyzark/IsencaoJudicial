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
from config import DOWNLOADS_DIR, MODEL_PATH, RESULTS_ISENCAO_DIR, ensure_parent

class ValidadorPassagens:
    def __init__(self, yolo_weights_path):
        self.yolo_model = YOLO(yolo_weights_path)
        self.yolo_model.to("cuda")
        self.ocr_reader = easyocr.Reader(['pt'], gpu=True)  # Use gpu=False se nao tiver placa de video dedicada.
        self.ocr_allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        self.qtd_imagens_por_passagem = 3
        self.max_diferencas_aprovacao = 3
        self.largura_minima_ocr = 600

        # Placa antiga: ABC1234. Mercosul: ABC1D23.
        self.regex_placa = re.compile(r"([A-Z]{3}\d[A-Z\d]\d{2}|[A-Z]{3}\d{4})")
        self.regex_data = re.compile(r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})")

    def _limpar_texto_ocr(self, texto):
        texto = re.sub(r"[^A-Z0-9]", "", texto.upper())
        
        # CorreÃ§Ã£o para placas de moto onde o OCR lÃª a linha de baixo primeiro (1234ABC)
        if re.fullmatch(r"\d{4}[A-Z]{3}", texto):
            texto = texto[4:] + texto[:4]
        # CorreÃ§Ã£o para motos padrÃ£o Mercosul (1D23ABC)
        elif re.fullmatch(r"\d[A-Z]\d{2}[A-Z]{3}", texto):
            texto = texto[4:] + texto[:4]
            
        return texto

    def _descobrir_mascara(self, placa):
        """Descobre se a placa original (Ground Truth) Ã© Antiga ou Mercosul para guiar o OCR."""
        placa_limpa = re.sub(r"[^A-Z0-9]", "", str(placa).upper())
        if not placa_limpa or len(placa_limpa) != 7:
            return None
        # Se o 5Âº caractere (Ã­ndice 4) for letra, Ã© Mercosul. SenÃ£o, antiga.
        if placa_limpa[4].isalpha():
            return "LLLDLDD"
        return "LLLDDDD"

    def _corrigir_por_mascara(self, texto, mascara):
        letras = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B", "7": "Z"}
        digitos = {
            "O": "0", "Q": "0", "D": "0",
            "I": "1", "L": "1", "T": "1",
            "Z": "2",
            "A": "4",
            "S": "5",
            "G": "6",
            "B": "8", "E": "8",
            "J": "1", 
            "U": "0", 
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

        # Se sabemos qual Ã© a placa base do PDF, testamos SÃ“ a mÃ¡scara dela.
        # SenÃ£o, tentamos as duas (fallback).
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

        # FIX: ordena por posiÃ§Ã£o â€” prefere o trecho mais Ã  direita do texto
        # porque o caractere extra costuma aparecer Ã  esquerda (faixa Mercosul)
        # Desempate: posiÃ§Ã£o mais alta (mais prÃ³xima do fim) ganha
        candidatos.sort(key=lambda x: -x[0])
        return candidatos[0][1]

    def _distancia_placas(self, placa_texto, placa_ocr):
        placa_texto = self._limpar_texto_ocr(placa_texto)
        placa_ocr = self._limpar_texto_ocr(placa_ocr)

        if not placa_texto or not placa_ocr:
            return 999

        # Expandido com mais vÃ­cios clÃ¡ssicos de OCR de placas brasileiras
        confundiveis = {
            ("0", "O"), ("O", "0"),
            ("1", "I"), ("I", "1"), ("1", "L"), ("L", "1"), ("T", "I"), ("I", "T"),
            ("2", "Z"), ("Z", "2"), ("Z", "7"), ("7", "Z"),
            ("4", "A"), ("A", "4"),
            ("5", "S"), ("S", "5"),
            ("6", "G"), ("G", "6"),
            ("8", "B"), ("B", "8"), ("B", "E"), ("E", "B"),
            ("D", "0"), ("0", "D"), ("D", "O"), ("O", "D")
        }

        anterior = list(range(len(placa_ocr) + 1))
        for i, char_texto in enumerate(placa_texto, start=1):
            atual = [i]
            for j, char_ocr in enumerate(placa_ocr, start=1):
                if char_texto == char_ocr:
                    custo_substituicao = 0
                elif (char_texto, char_ocr) in confundiveis:
                    custo_substituicao = 0.35
                else:
                    custo_substituicao = 1

                atual.append(min(
                    anterior[j] + 1,
                    atual[j - 1] + 1,
                    anterior[j - 1] + custo_substituicao,
                ))
            anterior = atual

        return anterior[-1]

    def _ler_ocr_com_cache(self, imagem_bytes, ocr_cache, mascara_esperada=None):
        if not imagem_bytes:
            return ""

        chave_cache = id(imagem_bytes)
        if chave_cache not in ocr_cache:
            ocr_cache[chave_cache] = self.processar_imagem_yolo_ocr(imagem_bytes, mascara_esperada)

        resultado = ocr_cache[chave_cache]
        if resultado == "Nao foi possivel identificar os caracteres":
            return ""
        return self._normalizar_placa_ocr(resultado, mascara_esperada)

    def avaliar_passagem(self, placa_texto, imagens_candidatas, ocr_cache):
        # Extrai qual o modelo de placa o PDF diz que Ã© para guiar o OCR
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

        melhor_placa = ""
        melhor_distancia = 999
        melhor_indice = None
        melhor_ocr_bruto = ""

        # Processa sempre todas as 3 imagens e elege a campeÃ£
        for indice, imagem_bytes in enumerate(imagens_candidatas):
            placa_ocr = self._ler_ocr_com_cache(imagem_bytes, ocr_cache, mascara_esperada)
            if not placa_ocr:
                continue

            distancia = self._distancia_placas(placa_texto_limpa, placa_ocr)
            
            if distancia < melhor_distancia:
                melhor_distancia = distancia
                melhor_placa = placa_ocr
                melhor_indice = indice
                melhor_ocr_bruto = placa_ocr

            # OtimizaÃ§Ã£o: se a distÃ¢ncia for perfeita (0), jÃ¡ pode parar de procurar
            if distancia == 0:
                break

        # Decide aprovaÃ§Ã£o baseado na melhor imagem encontrada do grupo
        if melhor_distancia <= self.max_diferencas_aprovacao:
            return {
                "placa_final": placa_texto_limpa,
                "ocr_bruto": melhor_ocr_bruto,
                "status": "Aprovado - busca nas imagens",
                "diferenca": melhor_distancia,
                "imagem_usada": (melhor_indice + 1) if melhor_indice is not None else "",
            }

        return {
            "placa_final": melhor_placa or "Nao foi possivel identificar os caracteres",
            "ocr_bruto": melhor_ocr_bruto,
            "status": "Revisar",
            "diferenca": melhor_distancia,
            "imagem_usada": (melhor_indice + 1) if melhor_indice is not None else "",
        }

    def _extrair_imagens_posicionadas(self, doc, page):
        imagens = []
        imagem_cache = {}

        for img in page.get_images(full=True):
            xref = img[0]
            largura_original = img[2]
            altura_original = img[3]

            if largura_original < 80 or altura_original < 40:
                continue

            rects = page.get_image_rects(xref)
            if not rects:
                continue

            if xref not in imagem_cache:
                base_image = doc.extract_image(xref)
                imagem_cache[xref] = base_image["image"]

            for rect in rects:
                if rect.width < 20 or rect.height < 20:
                    continue

                imagens.append({
                    "xref": xref,
                    "rect": rect,
                    "bytes": imagem_cache[xref],
                    "centro_x": (rect.x0 + rect.x1) / 2,
                    "centro_y": (rect.y0 + rect.y1) / 2,
                })

        return sorted(imagens, key=lambda item: (item["rect"].y0, item["rect"].x0))

    def _localizar_placa_na_pagina(self, page, placa_texto, y_cursor):
        rects = sorted(page.search_for(placa_texto), key=lambda rect: (rect.y0, rect.x0))

        for rect in rects:
            if rect.y0 >= y_cursor - 1:
                return rect

        return rects[0] if rects else None

    def _selecionar_imagens_por_posicao(self, passagens_pagina, imagens_pagina, indice_passagem):
        inicio = indice_passagem * self.qtd_imagens_por_passagem
        fim = inicio + self.qtd_imagens_por_passagem
        
        imagens_fatiadas = imagens_pagina[inicio:fim]
        return [img["bytes"] for img in imagens_fatiadas]

    def extrair_dados_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        passagens = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            imagens_pagina = self._extrair_imagens_posicionadas(doc, page)
            blocos_texto = page.get_text("text").split("Data/Hora")

            passagens_pagina = []
            y_cursor = -1
            for bloco in blocos_texto[1:]:
                try:
                    data_hora = self.regex_data.search(bloco).group(0)

                    linhas = bloco.split('\n')
                    categoria = [linha.strip() for linha in linhas if linha.strip().isdigit() and len(linha.strip()) <= 2][0]
                    placa_texto = self.regex_placa.search(bloco).group(0)

                    rect_placa = self._localizar_placa_na_pagina(page, placa_texto, y_cursor)
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

            for indice_passagem, passagem in enumerate(passagens_pagina):
                imagens_candidatas = self._selecionar_imagens_por_posicao(
                    passagens_pagina,
                    imagens_pagina,
                    indice_passagem,
                )
                passagens.append({
                    "Data/Hora": passagem["Data/Hora"],
                    "Categoria": passagem["Categoria"],
                    "Placa (Texto)": passagem["Placa (Texto)"],
                    "imagens_candidatas": imagens_candidatas,
                    "Pagina": page_num + 1,
                })

        return passagens

    def pre_processar_imagem(self, img_array):
        if img_array.size == 0:
            return []

        _, largura = img_array.shape[:2]
        escala = max(3.0, self.largura_minima_ocr / max(largura, 1))
        img_array = cv2.resize(img_array, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        contraste = clahe.apply(gray)

        blur = cv2.GaussianBlur(contraste, (3, 3), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, otsu_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adapt = cv2.adaptiveThreshold(contraste, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)

        kernel_engrossar = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        adapt = cv2.morphologyEx(adapt, cv2.MORPH_OPEN, kernel_engrossar) # Limpa sujeira pequena
        # Opcional: engrossar um pouco o preto (letras) se estiverem muito finas
        adapt = cv2.erode(adapt, kernel_engrossar, iterations=1)

        return [img_array, contraste, otsu, otsu_inv, adapt]

    def processar_imagem_yolo_ocr(self, imagem_bytes, mascara_esperada=None):
        if not imagem_bytes:
            return "Nao foi possivel identificar os caracteres"

        # Converte bytes para imagem OpenCV
        pil_image = Image.open(BytesIO(imagem_bytes)).convert("RGB")
        img_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 1. Executa a detecÃ§Ã£o do YOLO
        resultados_yolo = self.yolo_model(img_cv2, imgsz=1024, conf=0.25, iou=0.45, device=0, verbose=False)
        
        melhor_texto = ""
        melhor_score = -1

        for r in resultados_yolo:
            boxes = r.boxes
            # Ordena as caixas encontradas pela maior confianÃ§a
            boxes_ordenadas = sorted(
                boxes,
                key=lambda b: float(b.conf[0]) if b.conf is not None else 0,
                reverse=True,
            )

            for i_box, box in enumerate(boxes_ordenadas):
                conf_deteccao = float(box.conf[0])
                if conf_deteccao < 0.35: 
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                largura_box = x2 - x1
                altura_box = y2 - y1
                razao_proporcao = largura_box / max(altura_box, 1)
                
                # Filtro de proporÃ§Ã£o: 0.8 permite placas de moto (quadradas)
                if razao_proporcao < 0.8 or razao_proporcao > 6.0:
                    continue

                # Recorta a placa da imagem original
                recorte_placa = img_cv2[y1:y2, x1:x2]
                
                # Gera as 5 versÃµes processadas (Original esticada, Contraste, Otsu, Otsu_Inv, Adaptativo)
                imagens_para_ocr = self.pre_processar_imagem(recorte_placa)

                # Testa o OCR em cada uma das versÃµes de imagem geradas
                for imagem_ocr in imagens_para_ocr:
                    resultados_ocr = self.ocr_reader.readtext(
                        imagem_ocr,
                        detail=1,
                        allowlist=self.ocr_allowlist,
                        decoder='beamsearch',
                        beamWidth=10,
                        text_threshold=0.35,
                        mag_ratio=2

                    )

                    # Coleta candidatos (texto e confianÃ§a)
                    candidatos_ocr = [(res[1], float(res[2])) for res in resultados_ocr]
                    
                    # Se o OCR leu em blocos separados, tenta juntar tudo
                    if len(resultados_ocr) > 1:
                        texto_unido = "".join(res[1] for res in resultados_ocr)
                        conf_media = float(np.mean([res[2] for res in resultados_ocr]))
                        candidatos_ocr.append((texto_unido, conf_media))

                    for texto_bruto, confianca in candidatos_ocr:
                        # Normaliza e corrige usando a mÃ¡scara do PDF como guia[cite: 1]
                        texto = self._normalizar_placa_ocr(texto_bruto, mascara_esperada)
                        if not texto:
                            continue

                        placa_valida = self.regex_placa.fullmatch(texto) is not None
                        
                        # CÃ¡lculo de Score: bonifica placas com 7 caracteres e vÃ¡lidas pelo Regex[cite: 1]
                        score = confianca + (5.0 if placa_valida else 0.0) - abs(len(texto) - 7) * 2.0
                        
                        if score > melhor_score:
                            melhor_score = score
                            melhor_texto = texto

                        # Atalho: se achou uma leitura perfeita e confiÃ¡vel, jÃ¡ encerra[cite: 1]
                        if placa_valida and confianca >= 0.40:
                            return texto

        return melhor_texto if melhor_texto else "Nao foi possivel identificar os caracteres"
    
    def processar(self, caminho_entrada: str, caminho_saida: str):
        """
        Interface unificada compatível com main.py.
        Aceita tanto um arquivo .pdf único quanto uma pasta de .pdf.
        """
        from glob import glob

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
            passagens = self.extrair_dados_pdf(str(pdf))
            print(f'  {len(passagens)} passagem(ns) encontrada(s).')
            todas_passagens.extend(passagens)

        if not todas_passagens:
            print('\nNenhuma passagem encontrada nos arquivos.')
            return

        print(f'\nIniciando processamento YOLO + OCR ({len(todas_passagens)} passagens)...')
        self.gerar_relatorio(todas_passagens, caminho_saida)

    def gerar_relatorio(self, passagens, output_excel):
        dados_finais = []
        
        for p in tqdm(passagens, desc="Processando passagens"):
            ocr_cache = {}
            avaliacao = self.avaliar_passagem(
                p["Placa (Texto)"],
                p.get("imagens_candidatas", []),
                ocr_cache,
            )

            dados_finais.append({
                "Pagina": p.get("Pagina", ""),
                "Data/Hora": p["Data/Hora"],
                "Categoria": p["Categoria"],
                "Placa (Texto)": p["Placa (Texto)"],
                "Placa (Imagem/OCR)": avaliacao["placa_final"],
                "OCR Bruto": avaliacao["ocr_bruto"],
                "Status OCR": avaliacao["status"],
                "Diferenca OCR": avaliacao["diferenca"],
                "Imagem Usada": avaliacao["imagem_usada"],
                "Qtd Imagens Candidatas": len(p.get("imagens_candidatas", [])),
            })

        df = pd.DataFrame(dados_finais)
        df.to_excel(output_excel, index=False)
        print(f"Processamento concluido. Salvo em: {output_excel}")
        return df

if __name__ == "__main__":
    CAMINHO_YOLO_WEIGHTS = MODEL_PATH
    CAMINHO_PDF = DOWNLOADS_DIR / '01.pdf'
    SAIDA_EXCEL = ensure_parent(RESULTS_ISENCAO_DIR / 'resultado_autoban.xlsx')

    validador = ValidadorPassagens(yolo_weights_path=CAMINHO_YOLO_WEIGHTS)

    print("Extraindo dados do PDF...")
    passagens_extraidas = validador.extrair_dados_pdf(CAMINHO_PDF)

    print(f"Total de registros encontrados: {len(passagens_extraidas)}")
    print("Iniciando processamento YOLO + OCR...")

    df_resultado = validador.gerar_relatorio(passagens_extraidas, SAIDA_EXCEL)