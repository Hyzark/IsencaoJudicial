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
from pathlib import Path

# Ajuste os imports conforme a sua config.py
from config import DOWNLOADS_DIR, MODEL_PATH, RESULTS_ISENCAO_DIR, ensure_parent

class ValidadorPdfEixo:
    def __init__(self, yolo_weights_path):
        self.yolo_model = YOLO(yolo_weights_path)
        self.yolo_model.to("cuda")
        self.ocr_reader = easyocr.Reader(['pt'], gpu=True)
        self.ocr_allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        self.max_diferencas_aprovacao = 3
        self.largura_minima_ocr = 600

        # Regex para capturar os dados do PDF da Eixo SP
        self.regex_placa = re.compile(r"(?<![A-Z0-9])([A-Z]{3}\d[A-Z\d]\d{2}|[A-Z]{3}\d{4})(?![A-Z0-9])")
        self.regex_data_transito = re.compile(
            r"Data do transito.*?(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})",
            re.DOTALL
        )
        self.regex_id = re.compile(r"(\d+[A-Z]{2,3}\d{10,})")
    # ── HERANÇA DE LÓGICA DO OCR (Autoban) ───────────────────────────────────
    def _limpar_texto_ocr(self, texto):
        texto = re.sub(r"[^A-Z0-9]", "", texto.upper())
        if re.fullmatch(r"\d{4}[A-Z]{3}", texto):
            texto = texto[4:] + texto[:4]
        elif re.fullmatch(r"\d[A-Z]\d{2}[A-Z]{3}", texto):
            texto = texto[4:] + texto[:4]
        return texto

    def _descobrir_mascara(self, placa):
        placa_limpa = re.sub(r"[^A-Z0-9]", "", str(placa).upper())
        if not placa_limpa or len(placa_limpa) != 7: return None
        return "LLLDLDD" if placa_limpa[4].isalpha() else "LLLDDDD"

    def _corrigir_por_mascara(self, texto, mascara):
        letras = {"0":"O","1":"I","2":"Z","4":"A","5":"S","6":"G","8":"B","7":"Z"}
        digitos = {"O":"0","Q":"0","D":"0","I":"1","L":"1","T":"1","Z":"2","A":"4","S":"5","G":"6","B":"8","E":"8","J":"1","U":"0"}
        return "".join(letras.get(c, c) if m == "L" else digitos.get(c, c) for c, m in zip(texto, mascara))

    def _normalizar_placa_ocr(self, texto, mascara_esperada=None):
        texto = self._limpar_texto_ocr(texto)
        if not texto: return ""
        if self.regex_placa.fullmatch(texto): return texto

        mascaras = [mascara_esperada] if mascara_esperada else ["LLLDDDD", "LLLDLDD"]
        candidatos = []
        for inicio in range(max(0, len(texto) - 6)):
            trecho = texto[inicio:inicio + 7]
            if len(trecho) != 7: continue
            for mascara in mascaras:
                corrigido = self._corrigir_por_mascara(trecho, mascara)
                if self.regex_placa.fullmatch(corrigido):
                    candidatos.append((inicio, corrigido))
        
        if not candidatos: return texto[:7]
        candidatos.sort(key=lambda x: -x[0])
        return candidatos[0][1]

    def _distancia_placas(self, placa_texto, placa_ocr):
        placa_texto = self._limpar_texto_ocr(placa_texto)
        placa_ocr = self._limpar_texto_ocr(placa_ocr)
        if not placa_texto or not placa_ocr: return 999

        confundiveis = {("0","O"),("O","0"),("1","I"),("I","1"),("1","L"),("L","1"),("T","I"),("I","T"),("2","Z"),("Z","2"),("Z","7"),("7","Z"),("4","A"),("A","4"),("5","S"),("S","5"),("6","G"),("G","6"),("8","B"),("B","8"),("B","E"),("E","B"),("D","0"),("0","D"),("D","O"),("O","D")}
        anterior = list(range(len(placa_ocr) + 1))
        
        for i, char_texto in enumerate(placa_texto, start=1):
            atual = [i]
            for j, char_ocr in enumerate(placa_ocr, start=1):
                custo = 0 if char_texto == char_ocr else (0.35 if (char_texto, char_ocr) in confundiveis else 1)
                atual.append(min(anterior[j]+1, atual[j-1]+1, anterior[j-1]+custo))
            anterior = atual
        return anterior[-1]

    def _ler_ocr_com_cache(self, imagem_bytes, ocr_cache, mascara_esperada=None):
        if not imagem_bytes: return ""
        chave_cache = id(imagem_bytes)
        if chave_cache not in ocr_cache:
            ocr_cache[chave_cache] = self.processar_imagem_yolo_ocr(imagem_bytes, mascara_esperada)
        res = ocr_cache[chave_cache]
        return "" if res == "Não foi possível identificar" else self._normalizar_placa_ocr(res, mascara_esperada)

    def avaliar_passagem(self, placa_texto, imagens_candidatas, ocr_cache):
        mascara_esperada = self._descobrir_mascara(placa_texto)
        placa_texto_limpa = self._normalizar_placa_ocr(placa_texto, mascara_esperada)
        
        if not imagens_candidatas:
            return {"placa_final": "Não foi possível identificar", "ocr_bruto": "", "status": "Revisar - sem imagem", "diferenca": 999, "imagem_usada": ""}

        melhor_placa, melhor_distancia, melhor_indice, melhor_ocr_bruto = "", 999, None, ""

        for indice, imagem_bytes in enumerate(imagens_candidatas):
            placa_ocr = self._ler_ocr_com_cache(imagem_bytes, ocr_cache, mascara_esperada)
            if not placa_ocr: continue

            distancia = self._distancia_placas(placa_texto_limpa, placa_ocr)
            if distancia < melhor_distancia:
                melhor_distancia, melhor_placa, melhor_indice, melhor_ocr_bruto = distancia, placa_ocr, indice, placa_ocr
            if distancia == 0: break

        status = "Aprovado - busca nas imagens" if melhor_distancia <= self.max_diferencas_aprovacao else "Revisar"
        return {
            "placa_final": melhor_placa or "Não foi possível identificar",
            "ocr_bruto": melhor_ocr_bruto,
            "status": status,
            "diferenca": melhor_distancia,
            "imagem_usada": (melhor_indice + 1) if melhor_indice is not None else ""
        }

    def pre_processar_imagem(self, img_array):
        if img_array.size == 0: return []
        escala = max(3.0, self.largura_minima_ocr / max(img_array.shape[1], 1))
        img_array = cv2.resize(img_array, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        contraste = clahe.apply(gray)
        blur = cv2.GaussianBlur(contraste, (3, 3), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, otsu_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adapt = cv2.adaptiveThreshold(contraste, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        adapt = cv2.morphologyEx(adapt, cv2.MORPH_OPEN, kernel)
        adapt = cv2.erode(adapt, kernel, iterations=1)
        return [img_array, contraste, otsu, otsu_inv, adapt]

    def processar_imagem_yolo_ocr(self, imagem_bytes, mascara_esperada=None):
        if not imagem_bytes: return "Não foi possível identificar"
        pil_image = Image.open(BytesIO(imagem_bytes)).convert("RGB")
        img_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        resultados_yolo = self.yolo_model(img_cv2, imgsz=640, conf=0.25, iou=0.45, device=0, verbose=False)
        melhor_texto, melhor_score = "", -1

        for r in resultados_yolo:
            boxes_ordenadas = sorted(r.boxes, key=lambda b: float(b.conf[0]) if b.conf is not None else 0, reverse=True)
            for box in boxes_ordenadas[:1]:
                if float(box.conf[0]) < 0.35: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                razao = (x2 - x1) / max((y2 - y1), 1)
                if razao < 0.8 or razao > 6.0: continue

                imagens_para_ocr = self.pre_processar_imagem(img_cv2[y1:y2, x1:x2])
                for imagem_ocr in imagens_para_ocr:
                    resultados_ocr = self.ocr_reader.readtext(imagem_ocr, detail=0, allowlist=self.ocr_allowlist, decoder='greedy', text_threshold=0.35)
                    candidatos = [(res[1], float(res[2])) for res in resultados_ocr]
                    if len(resultados_ocr) > 1:
                        candidatos.append(("".join(res[1] for res in resultados_ocr), float(np.mean([res[2] for res in resultados_ocr]))))

                    for texto_bruto, conf in candidatos:
                        texto = self._normalizar_placa_ocr(texto_bruto, mascara_esperada)
                        if not texto: continue
                        valida = self.regex_placa.fullmatch(texto) is not None
                        score = conf + (5.0 if valida else 0.0) - abs(len(texto) - 7) * 2.0
                        if score > melhor_score: melhor_score, melhor_texto = score, texto
                        if valida and conf >= 0.40: return texto

        return melhor_texto if melhor_texto else "Não foi possível identificar"

    # ── EXTRAÇÃO ESPECÍFICA DO PDF EIXO ──────────────────────────────────────
    def extrair_dados_pdf(self, pdf_path):
            doc = fitz.open(pdf_path)
            passagens = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                # Extraímos o texto limpando as aspas que o PDF da Eixo costuma trazer
                texto_bruto = page.get_text("text").replace('"', '')
                linhas = [l.strip() for l in texto_bruto.split('\n') if l.strip()]

                id_match = self.regex_id.search(texto_bruto)
                
                if id_match:
                    id_transacao = id_match.group(1)
                    
                    # LÓGICA DE COLUNA: 
                    # No PDF da Eixo, a Placa geralmente aparece algumas linhas DEPOIS do ID
                    # Vamos localizar o índice do ID na lista de linhas
                    try:
                        idx_id = linhas.index(id_transacao)
                        # A estrutura comum é: [ID], [Categoria], [Placa]
                        # Então a categoria está em idx+1 e a placa em idx+2
                        categoria = linhas[idx_id + 1]
                        placa_texto = linhas[idx_id + 2]
                        
                        # Validamos se o que achamos na "coluna" realmente parece uma placa
                        if not self.regex_placa.fullmatch(placa_texto):
                            # Se falhar, tentamos buscar na página toda com o regex rigoroso
                            placa_match = self.regex_placa.search(texto_bruto)
                            placa_texto = placa_match.group(0) if placa_match else "N/A"
                    except (ValueError, IndexError):
                        # Fallback caso a estrutura de linhas mude
                        placa_match = self.regex_placa.search(texto_bruto)
                        placa_texto = placa_match.group(0) if placa_match else "N/A"
                        categoria = "01"

                    # Busca a data na página
                    data_match = self.regex_data_transito.search(texto_bruto)
                    data_hora = data_match.group(1) if data_match else "00/00/0000 00:00:00"

                    # Extração de imagens da página (mesma lógica)
                    imagens_candidatas = []
                    for img_info in page.get_images(full=True):
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        if base_image["width"] > 100: # Ignora ícones
                            imagens_candidatas.append(base_image["image"])

                    passagens.append({
                        "ID Extraído": id_transacao,
                        "Data/Hora": data_hora,
                        "Categoria": categoria,
                        "Placa (Texto)": placa_texto,
                        "imagens_candidatas": imagens_candidatas,
                        "Pagina": page_num + 1,
                    })

            return passagens

    def processar(self, caminho_entrada: str, caminho_saida: str):
        caminho = Path(caminho_entrada)
        arquivos = list(caminho.glob('*.pdf')) if caminho.is_dir() else [caminho] if caminho.is_file() else []

        if not arquivos:
            print('Nenhum arquivo .pdf encontrado.')
            return

        todas_passagens = []
        for pdf in arquivos:
            print(f'\nExtraindo dados de: {pdf.name}')
            passagens = self.extrair_dados_pdf(str(pdf))
            print(f'  {len(passagens)} passagem(ns) encontrada(s) (1 por página).')
            todas_passagens.extend(passagens)

        if not todas_passagens: return
        print(f'\nIniciando processamento YOLO + OCR ({len(todas_passagens)} passagens)...')
        self.gerar_relatorio(todas_passagens, caminho_saida)

    # ── GERAÇÃO DE RELATÓRIO DUPLO & TERMINAL (Padrão Tamoios) ───────────────
    def gerar_relatorio(self, passagens, output_excel):
        dados_finais = []
        
        for p in tqdm(passagens, desc="Processando imagens"):
            ocr_cache = {}
            avaliacao = self.avaliar_passagem(p["Placa (Texto)"], p.get("imagens_candidatas", []), ocr_cache)

            dados_finais.append({
                "ID Extraído": p["ID Extraído"],
                "Pagina PDF": p["Pagina"],
                "Data/Hora": p["Data/Hora"],
                "Categoria": p["Categoria"],
                "Placa Planilha": p["Placa (Texto)"], # Mudei o nome para ficar padronizado
                "Placa OCR": avaliacao["placa_final"],
                "OCR Bruto": avaliacao["ocr_bruto"],
                "Status OCR": avaliacao["status"],
                "Diferença OCR": avaliacao["diferenca"],
                "Imagem Usada": avaliacao["imagem_usada"],
                "Total Imagens": len(p.get("imagens_candidatas", [])),
            })

        df_completo = pd.DataFrame(dados_finais)

        # ── Lógica de Deduplicação ──
        df_completo['Diferença Numérica'] = pd.to_numeric(df_completo['Diferença OCR'], errors='coerce')
        prioridade_status = {
            'Aprovado - busca nas imagens': 0,
            'Revisar': 1,
            'Revisar - sem imagem': 2,
            'ID não encontrado na planilha': 3,
            'ID fora do padrão': 4
        }
        df_completo['Prioridade'] = df_completo['Status OCR'].map(prioridade_status).fillna(5)

        # ── Salva em Múltiplas Abas ──
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            df_completo.to_excel(writer, sheet_name='Processamento Completo', index=False)
        # ── Resumo de Terminal ──
        total_id_unicos = len(df_completo)
        aprovados = (df_completo['Status OCR'] == 'Aprovado - busca nas imagens').sum()
        revisar = (df_completo['Status OCR'] == 'Revisar').sum()
        sem_img = (df_completo['Status OCR'] == 'Revisar - sem imagem').sum()
        
        print(f"\n{'='*54}")
        print(f"  RELATÓRIO CONSOLIDADO EIXO SP")
        print(f"  Total de IDs únicos:  {total_id_unicos}")
        print(f"  APROVADOS: {aprovados:3d} ({aprovados/total_id_unicos*100:.2f}%)")
        print(f"  REVISAR OCR: {revisar:3d} ({revisar/total_id_unicos*100:.2f}%)")
        print(f"  SEM IMAGEM: {sem_img:3d} ({sem_img/total_id_unicos*100:.2f}%)")
        print(f"  ─────────────────────────────────────────")
        print(f"  Total de passagens lidas: {len(df_completo)}")
        print(f"{'='*54}")
        print(f"  Arquivo gerado: {output_excel}")

if __name__ == "__main__":
    CAMINHO_YOLO_WEIGHTS = MODEL_PATH
    CAMINHO_PDF = DOWNLOADS_DIR / 'Isentos012023.pdf' 
    # Usando a pasta de eixo como combinamos na config.py
    SAIDA_EXCEL = ensure_parent(RESULTS_ISENCAO_DIR / 'resultado_eixosp.xlsx')

    validador = ValidadorPdfEixo(yolo_weights_path=CAMINHO_YOLO_WEIGHTS)
    validador.processar(CAMINHO_PDF, SAIDA_EXCEL)