import fitz  # PyMuPDF
import cv2
import easyocr
import pandas as pd
import numpy as np
from ultralytics import YOLO
import re
from io import BytesIO
from PIL import Image
from tqdm import tqdm  # Importação da barra de progresso

class ValidadorPassagens:
    def __init__(self, yolo_weights_path):
        # Inicializa os modelos
        self.yolo_model = YOLO(yolo_weights_path)
        self.ocr_reader = easyocr.Reader(['pt'], gpu=True) # Mude para gpu=False se não tiver placa de vídeo dedicada no ambiente local
        
        # Regex para identificar o padrão de Data/Hora, Categoria e Placa (Padrão Antigo e Mercosul)
        self.regex_placa = re.compile(r"([A-Z]{3}\d[A-Z\d]\d{2}|[A-Z]{3}\d{4})")
        self.regex_data = re.compile(r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})")

    def extrair_dados_pdf(self, pdf_path):
        """
        Lê o PDF, extrai os textos e associa às imagens baseado na ordem/posição.
        """
        doc = fitz.open(pdf_path)
        passagens = []

        # Barra de progresso para a leitura das páginas do PDF
        for page_num in tqdm(range(len(doc)), desc="Extraindo dados do PDF"):
            page = doc[page_num]
            
            # Extrai blocos de texto e imagens
            text_dict = page.get_text("dict")
            images = page.get_images(full=True)
            
            # Heurística simplificada: busca textos e imagens sequencialmente
            # Em PDFs da concessionária, a estrutura costuma ser sequencial por bloco
            blocos_texto = page.get_text("text").split("Data/Hora")
            
            img_index = 0
            for bloco in blocos_texto[1:]: # Ignora o cabeçalho
                try:
                    data_hora = self.regex_data.search(bloco).group(0)
                    
                    # Extração da Categoria (logo após a data) e Placa
                    linhas = bloco.split('\n')
                    categoria = [linha.strip() for linha in linhas if linha.strip().isdigit() and len(linha.strip()) <= 2][0]
                    placa_texto = self.regex_placa.search(bloco).group(0)
                    
                    # Associa a próxima imagem disponível no PDF a esta passagem
                    imagem_bytes = None
                    if img_index < len(images):
                        xref = images[img_index][0]
                        base_image = doc.extract_image(xref)
                        imagem_bytes = base_image["image"]
                        img_index += 1 # Avança para a próxima imagem (pode precisar de ajuste se houver múltiplas fotos por passagem)

                    passagens.append({
                        "Data/Hora": data_hora,
                        "Categoria": categoria,
                        "Placa (Texto)": placa_texto,
                        "imagem_bytes": imagem_bytes
                    })
                except AttributeError:
                    continue # Pula blocos que não casam perfeitamente com o Regex

        return passagens

    def pre_processar_imagem(self, img_array):
        """
        Aplica filtros na imagem recortada da placa para lidar com ruído noturno e brilho de farol.
        """
        # Converte para escala de cinza
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # Filtro Bilateral: reduz ruído mantendo as bordas nítidas (ótimo para caracteres)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17) 
        
        # Equalização de histograma adaptativa (CLAHE) para corrigir brilho do farol
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(bfilter)
        
        # Threshold adaptativo
        thresh = cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        return thresh

    def processar_imagem_yolo_ocr(self, imagem_bytes):
        """
        Recebe a imagem bruta, detecta a placa, recorta, trata e lê o texto.
        """
        if not imagem_bytes:
            return "Não foi possível identificar os caracteres"

        # Converte bytes para array do OpenCV
        pil_image = Image.open(BytesIO(imagem_bytes))
        img_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 1. Inferência YOLO
        resultados_yolo = self.yolo_model(img_cv2)
        
        for r in resultados_yolo:
            boxes = r.boxes
            if len(boxes) == 0:
                continue # Nenhuma placa detectada pelo YOLO
                
            # Pega a primeira placa detectada (maior confiança)
            box = boxes[0] 
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Recorta a região da placa
            recorte_placa = img_cv2[y1:y2, x1:x2]
            
            # 2. Pré-processamento
            placa_tratada = self.pre_processar_imagem(recorte_placa)
            
            # 3. OCR
            resultados_ocr = self.ocr_reader.readtext(placa_tratada, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            
            if resultados_ocr:
                # Junta o texto caso o OCR separe letras e números
                texto_final = "".join(resultados_ocr).replace(" ", "")
                return texto_final

        return "Não foi possível identificar os caracteres"

    def gerar_relatorio(self, passagens, output_excel):
        """
        Executa a pipeline completa e exporta o DataFrame.
        """
        dados_finais = []
        
        # Barra de progresso para o processamento do YOLO + OCR
        for p in tqdm(passagens, desc="Processando YOLO e OCR"):
            placa_ocr = self.processar_imagem_yolo_ocr(p['imagem_bytes'])
            
            dados_finais.append({
                "Data/Hora": p["Data/Hora"],
                "Categoria": p["Categoria"],
                "Placa (Texto)": p["Placa (Texto)"],
                "Placa (Imagem/OCR)": placa_ocr
            })
            
        df = pd.DataFrame(dados_finais)
        df.to_excel(output_excel, index=False)
        print(f"\nProcessamento concluído. Salvo em: {output_excel}")
        return df

# ==========================================
# Exemplo de Uso
# ==========================================
if __name__ == "__main__":
    # Substitua pelo caminho do peso treinado no Colab
    CAMINHO_YOLO_WEIGHTS = "best.pt" 
    CAMINHO_PDF = "01_02_2025-páginas-1.pdf"
    SAIDA_EXCEL = "validacao_passagens.xlsx"

    validador = ValidadorPassagens(yolo_weights_path=CAMINHO_YOLO_WEIGHTS)
    
    # Executa a pipeline
    print("\nIniciando a extração do PDF...")
    passagens_extraidas = validador.extrair_dados_pdf(CAMINHO_PDF)
    
    print(f"\nTotal de registros encontrados: {len(passagens_extraidas)}")
    print("Iniciando processamento de Imagens (YOLO + OCR)...")
    
    df_resultado = validador.gerar_relatorio(passagens_extraidas, SAIDA_EXCEL)