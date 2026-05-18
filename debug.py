"""
debug_ocr_visual.py
===================
Ferramenta de DEBUG visual para inspecionar cada etapa do pipeline OCR.

Gera um relatÃ³rio HTML interativo mostrando, para cada imagem processada:
  1. Imagem original completa (com bounding boxes do YOLO desenhados)
  2. Recorte bruto de cada detecÃ§Ã£o YOLO
  3. As 5 variantes de prÃ©-processamento geradas por _pre_processar_imagem
  4. O texto OCR lido em cada variante + confianÃ§a + score final
  5. Destaque visual na variante/recorte eleito como melhor resultado

Uso
----
  python debug_ocr_visual.py --pdf  caminho/para/arquivo.pdf
  python debug_ocr_visual.py --xlsx caminho/para/arquivo.xlsx
  python debug_ocr_visual.py --pdf  pasta/com/pdfs/     --todos
  python debug_ocr_visual.py --xlsx pasta/com/excels/   --todos

  --saida resultado_debug.html   (opcional; padrÃ£o: debug_ocr_<timestamp>.html)
  --yolo  caminho/para/best.pt   (opcional se definido em PATH_YOLO abaixo)
  --sem-gpu                      (forÃ§a CPU)

Resultado
----------
  Um Ãºnico arquivo .html autocontido (imagens em base64) que pode ser
  aberto em qualquer navegador, sem dependÃªncia de servidor.
"""

import argparse
import base64
import os
import re
import sys
from datetime import datetime
from glob import glob
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from config import MODEL_PATH

# â”€â”€ Caminho padrÃ£o do modelo YOLO (edite aqui para evitar passar --yolo) â”€â”€â”€â”€â”€â”€
PATH_YOLO_PADRAO = str(MODEL_PATH)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Helpers de imagem
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

NOMES_VARIANTES = [
    'Original escalada',
    'CLAHE (contraste local)',
    'Otsu normal',
    'Otsu invertido',
    'Threshold adaptativo',
]


def _array_para_base64(img: np.ndarray, formato: str = 'PNG') -> str:
    """Converte array OpenCV/numpy em string base64 para embed no HTML."""
    # Garante que imagens em escala de cinza ficam visÃ­veis
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pil = Image.fromarray(img_rgb)
    buf = BytesIO()
    pil.save(buf, format=formato)
    return base64.b64encode(buf.getvalue()).decode()


def _bytes_para_base64(raw: bytes, ext: str = 'png') -> str:
    """Bytes brutos de imagem â†’ base64."""
    return base64.b64encode(raw).decode()


def _desenhar_boxes(img_cv2: np.ndarray, boxes_info: list[dict]) -> np.ndarray:
    """Desenha bounding boxes coloridas sobre a imagem original."""
    img = img_cv2.copy()
    for i, b in enumerate(boxes_info):
        x1, y1, x2, y2 = b['x1'], b['y1'], b['x2'], b['y2']
        conf = b['conf']
        aceita = b['aceita']
        cor = (0, 200, 80) if aceita else (60, 60, 220)   # verde / vermelho
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, 3)
        label = f"#{i+1}  conf={conf:.2f}  razÃ£o={b['razao']:.2f}"
        if not aceita:
            label += f"  âœ— {b['motivo_rejeicao']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), cor, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return img


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Subclasse de debug â€” intercepta o pipeline sem alterar app_excel.py
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class DebugValidador:
    """
    Reimplementa _processar_imagem_yolo_ocr com coleta completa de dados
    de diagnÃ³stico.  NÃƒO herda ValidadorExcelPro para nÃ£o exigir GPU/YOLO
    durante importaÃ§Ã£o; recebe a instÃ¢ncia jÃ¡ criada pelo usuÃ¡rio.
    """

    def __init__(self, validador):
        """
        validador: instÃ¢ncia jÃ¡ inicializada de ValidadorExcelPro
                   (ou qualquer subclasse como ValidadorPDFNovoFormato).
        """
        self.v = validador   # acesso a yolo_model, ocr_reader, mÃ©todos helpers

    def processar_imagem_com_debug(
        self,
        imagem_bytes: bytes,
        placa_esperada: str,
        idx_imagem: int,
    ) -> dict:
        """
        VersÃ£o instrumentada de _processar_imagem_yolo_ocr.
        Coleta todas as etapas e retorna um dicionÃ¡rio rico para o relatÃ³rio.
        """
        v = self.v
        mascara_esperada = v._descobrir_mascara(placa_esperada)

        if not imagem_bytes:
            return {'erro': 'bytes vazio'}

        pil_img = Image.open(BytesIO(imagem_bytes)).convert('RGB')
        img_cv2 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # â”€â”€ YOLO â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        resultados_yolo = v.yolo_model(
            img_cv2, imgsz=1024, conf=0.25, iou=0.45,
            device=0 if v.yolo_model.device.type == 'cuda' else 'cpu',
            verbose=False,
        )

        boxes_info   = []   # metadados de cada box (para desenho)
        deteccoes    = []   # dados ricos de cada detecÃ§Ã£o aceita

        melhor_score  = -1
        melhor_texto  = ''
        melhor_det_i  = None
        melhor_var_i  = None

        for r in resultados_yolo:
            boxes_ordenadas = sorted(
                r.boxes,
                key=lambda b: float(b.conf[0]) if b.conf is not None else 0,
                reverse=True,
            )

            for box in boxes_ordenadas:
                conf_det = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                larg   = x2 - x1
                alt    = y2 - y1
                razao  = larg / max(alt, 1)

                aceita = conf_det >= 0.35 and 0.8 <= razao <= 6.0
                motivo = ''
                if conf_det < 0.35:
                    motivo = f'conf < 0.35'
                elif razao < 0.8:
                    motivo = f'razÃ£o {razao:.2f} < 0.8 (muito quadrado)'
                elif razao > 6.0:
                    motivo = f'razÃ£o {razao:.2f} > 6.0 (muito largo)'

                boxes_info.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'conf': conf_det, 'razao': razao,
                    'aceita': aceita, 'motivo_rejeicao': motivo,
                })

                if not aceita:
                    continue

                # Garante limites válidos
                h, w = img_cv2.shape[:2]

                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h))

                # Descarta bbox inválida
                if x2 <= x1 or y2 <= y1:
                    continue

                recorte = img_cv2[y1:y2, x1:x2]

                # Descarta recorte vazio
                if recorte is None or recorte.size == 0:
                    continue

                # Descarta recorte pequeno demais
                rh, rw = recorte.shape[:2]
                if rw < 10 or rh < 10:
                    continue

                variantes = v._pre_processar_imagem(recorte)

                variantes_debug = []

                for vi, variante in enumerate(variantes):
                    resultados_ocr = v.ocr_reader.readtext(
                        variante,
                        detail=1,
                        allowlist=v.ocr_allowlist,
                        decoder='beamsearch',
                        beamWidth=10,
                        text_threshold=0.35,
                    )

                    candidatos = [(res[1], float(res[2])) for res in resultados_ocr]
                    if len(resultados_ocr) > 1:
                        texto_unido = ''.join(res[1] for res in resultados_ocr)
                        conf_media  = float(np.mean([res[2] for res in resultados_ocr]))
                        candidatos.append((texto_unido, conf_media))

                    melhor_cand_texto = ''
                    melhor_cand_conf  = 0.0
                    melhor_cand_score = -1
                    ocr_detalhes      = []

                    for texto_bruto, confianca in candidatos:
                        texto        = v._normalizar_placa_ocr(texto_bruto, mascara_esperada)
                        placa_valida = v.regex_placa.fullmatch(texto) is not None
                        score        = (confianca
                                        + (5.0 if placa_valida else 0.0)
                                        - abs(len(texto) - 7) * 2.0)

                        ocr_detalhes.append({
                            'bruto':    texto_bruto,
                            'norm':     texto,
                            'conf':     round(confianca, 3),
                            'valida':   placa_valida,
                            'score':    round(score, 3),
                        })

                        if score > melhor_cand_score:
                            melhor_cand_score = score
                            melhor_cand_texto = texto
                            melhor_cand_conf  = confianca

                        # Atualiza melhor global
                        if score > melhor_score:
                            melhor_score  = score
                            melhor_texto  = texto
                            melhor_det_i  = len(deteccoes)
                            melhor_var_i  = vi

                    variantes_debug.append({
                        'nome':        NOMES_VARIANTES[vi],
                        'img_b64':     _array_para_base64(variante),
                        'ocr_det':     ocr_detalhes,
                        'melhor_text': melhor_cand_texto,
                        'melhor_conf': round(melhor_cand_conf, 3),
                        'melhor_score':round(melhor_cand_score, 3),
                    })

                deteccoes.append({
                    'box':       {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                  'conf': round(conf_det, 3), 'razao': round(razao, 2)},
                    'recorte_b64': _array_para_base64(recorte),
                    'variantes':   variantes_debug,
                })

        # Imagem original com boxes desenhadas
        img_anotada  = _desenhar_boxes(img_cv2, boxes_info)
        img_orig_b64 = _array_para_base64(img_cv2)
        img_anot_b64 = _array_para_base64(img_anotada)

        return {
            'idx':           idx_imagem,
            'placa_esperada':placa_esperada,
            'placa_ocr':     melhor_texto,
            'total_boxes':   len(boxes_info),
            'boxes_aceitas': sum(1 for b in boxes_info if b['aceita']),
            'img_orig_b64':  img_orig_b64,
            'img_anot_b64':  img_anot_b64,
            'deteccoes':     deteccoes,
            'melhor_det_i':  melhor_det_i,
            'melhor_var_i':  melhor_var_i,
            'melhor_score':  round(melhor_score, 3),
        }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Gerador de HTML
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _gerar_html(registros: list[dict], caminho_saida: str):
    """Gera o relatÃ³rio HTML completo com todas as imagens em base64."""

    cards_html = ''

    for reg in registros:
        arquivo   = reg['arquivo']
        passagem  = reg['passagem']
        imgs_debug= reg['imgs_debug']
        placa_esp = passagem.get('placa', '???')

        # â”€â”€ CabeÃ§alho do arquivo â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        cards_html += f'''
        <div class="arquivo-bloco">
          <div class="arquivo-header">
            <span class="arquivo-nome">ðŸ“„ {arquivo}</span>
            <span class="placa-badge">Placa esperada: <strong>{placa_esp}</strong></span>
            <span class="meta">ID: {passagem.get("id","â€”")} &nbsp;|&nbsp;
                               Data: {passagem.get("data","â€”")} &nbsp;|&nbsp;
                               Hora: {passagem.get("hora","â€”")}</span>
          </div>
        '''

        for img_d in imgs_debug:
            idx           = img_d['idx']
            placa_ocr     = img_d['placa_ocr'] or 'â€”'
            total_b       = img_d['total_boxes']
            aceitas       = img_d['boxes_aceitas']
            melhor_det_i  = img_d['melhor_det_i']
            melhor_var_i  = img_d['melhor_var_i']
            melhor_score  = img_d['melhor_score']

            match_class = 'match-ok' if placa_ocr == placa_esp else (
                'match-parcial' if placa_ocr not in ('â€”', 'Nao foi possivel identificar os caracteres') else 'match-fail'
            )

            cards_html += f'''
          <div class="img-bloco">
            <div class="img-bloco-header">
              <span class="img-idx">ðŸ“· Imagem #{idx+1}</span>
              <span class="ocr-resultado {match_class}">
                OCR: <strong>{placa_ocr}</strong>
              </span>
              <span class="score-badge">Score: {melhor_score}</span>
              <span class="boxes-info">{aceitas}/{total_b} detecÃ§Ãµes aceitas pelo YOLO</span>
            </div>

            <div class="originais-row">
              <div class="img-card orig">
                <div class="img-label">Original</div>
                <img src="data:image/png;base64,{img_d["img_orig_b64"]}" />
              </div>
              <div class="img-card orig">
                <div class="img-label">Com bounding boxes YOLO</div>
                <img src="data:image/png;base64,{img_d["img_anot_b64"]}" />
              </div>
            </div>
            '''

            if not img_d['deteccoes']:
                cards_html += '<div class="sem-det">âš ï¸ Nenhuma detecÃ§Ã£o aceita pelo YOLO nesta imagem.</div>'
            else:
                for di, det in enumerate(img_d['deteccoes']):
                    b = det['box']
                    is_best_det = (di == melhor_det_i)
                    cards_html += f'''
            <div class="det-bloco {"det-melhor" if is_best_det else ""}">
              <div class="det-header">
                {'â­ MELHOR DETECÃ‡ÃƒO &nbsp;|&nbsp; ' if is_best_det else ''}
                DetecÃ§Ã£o #{di+1} &nbsp;|&nbsp;
                bbox: ({b["x1"]},{b["y1"]})â†’({b["x2"]},{b["y2"]}) &nbsp;|&nbsp;
                conf: {b["conf"]} &nbsp;|&nbsp; razÃ£o: {b["razao"]}
              </div>
              <div class="recorte-row">
                <div class="img-card recorte">
                  <div class="img-label">Recorte YOLO</div>
                  <img src="data:image/png;base64,{det["recorte_b64"]}" />
                </div>
              </div>
              <div class="variantes-grid">
            '''

                    for vi, var in enumerate(det['variantes']):
                        is_best_var = is_best_det and (vi == melhor_var_i)
                        cands_html = ''
                        for c in var['ocr_det']:
                            cls = 'cand-valida' if c['valida'] else 'cand-invalida'
                            cands_html += f'''
                    <div class="candidato {cls}">
                      <span class="cand-bruto">bruto: {c["bruto"]}</span>
                      <span class="cand-norm">â†’ {c["norm"]}</span>
                      <span class="cand-meta">conf={c["conf"]} score={c["score"]}{"  âœ“ placa vÃ¡lida" if c["valida"] else ""}</span>
                    </div>'''

                        if not cands_html:
                            cands_html = '<div class="sem-ocr">OCR nÃ£o encontrou texto nesta variante.</div>'

                        cards_html += f'''
                <div class="var-card {"var-melhor" if is_best_var else ""}">
                  <div class="var-label">
                    {"â­ " if is_best_var else ""}
                    <strong>{vi}. {var["nome"]}</strong>
                    {"&nbsp;â† USADA" if is_best_var else ""}
                  </div>
                  <img src="data:image/png;base64,{var["img_b64"]}" />
                  <div class="var-ocr-resultado">
                    Leitura: <strong>{var["melhor_text"] or "â€”"}</strong>
                    &nbsp; conf={var["melhor_conf"]} &nbsp; score={var["melhor_score"]}
                  </div>
                  <div class="candidatos-lista">{cands_html}</div>
                </div>'''

                    cards_html += '</div></div>'   # fecha variantes-grid e det-bloco

            cards_html += '</div>'  # fecha img-bloco

        cards_html += '</div>'  # fecha arquivo-bloco

    # â”€â”€ Template HTML â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    html = f'''<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Debug OCR Visual â€” {datetime.now().strftime("%d/%m/%Y %H:%M")}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;800&display=swap');

  :root {{
    --bg:        #0d0f14;
    --bg2:       #13161e;
    --bg3:       #1a1e2a;
    --border:    #252a38;
    --accent:    #00e5ff;
    --green:     #00ff9d;
    --yellow:    #ffd600;
    --red:       #ff4444;
    --muted:     #5a6080;
    --text:      #ccd3f0;
    --text-dim:  #7a84a8;
    --radius:    8px;
    --mono:      'JetBrains Mono', monospace;
    --sans:      'Syne', sans-serif;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    font-size: 13px;
    padding: 24px;
  }}

  h1 {{
    font-family: var(--sans);
    font-size: 28px;
    font-weight: 800;
    color: var(--accent);
    letter-spacing: -0.5px;
    margin-bottom: 4px;
  }}
  .subtitle {{
    color: var(--text-dim);
    margin-bottom: 32px;
    font-size: 12px;
  }}

  /* â”€â”€ Arquivo â”€â”€ */
  .arquivo-bloco {{
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 40px;
    background: var(--bg2);
  }}
  .arquivo-header {{
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
    padding: 14px 20px;
    background: var(--bg3);
    border-bottom: 1px solid var(--border);
  }}
  .arquivo-nome {{ font-weight: 700; color: var(--accent); font-size: 14px; }}
  .placa-badge {{
    background: #00e5ff18;
    border: 1px solid var(--accent);
    border-radius: 6px;
    padding: 3px 10px;
    color: var(--accent);
    font-size: 13px;
  }}
  .meta {{ color: var(--text-dim); font-size: 11px; }}

  /* â”€â”€ Imagem bloco â”€â”€ */
  .img-bloco {{
    padding: 20px;
    border-bottom: 1px solid var(--border);
  }}
  .img-bloco:last-child {{ border-bottom: none; }}

  .img-bloco-header {{
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    margin-bottom: 16px;
  }}
  .img-idx {{ font-weight: 700; font-size: 14px; color: var(--yellow); }}

  .ocr-resultado {{ font-size: 14px; font-family: var(--mono); }}
  .match-ok    {{ color: var(--green); }}
  .match-parcial{{ color: var(--yellow); }}
  .match-fail  {{ color: var(--red); }}

  .score-badge {{
    background: #ffffff0d;
    border-radius: 4px;
    padding: 2px 8px;
    color: var(--text-dim);
    font-size: 11px;
  }}
  .boxes-info {{ font-size: 11px; color: var(--text-dim); }}

  /* â”€â”€ Linhas de imagens â”€â”€ */
  .originais-row {{
    display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px;
  }}
  .img-card {{
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    flex: 1; min-width: 280px;
  }}
  .img-card.orig img   {{ width: 100%; display: block; }}
  .img-card.recorte img{{ max-height: 120px; display: block; }}
  .img-label {{
    padding: 6px 12px;
    font-size: 11px;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    background: #ffffff06;
  }}

  /* â”€â”€ DetecÃ§Ã£o â”€â”€ */
  .det-bloco {{
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 16px;
    overflow: hidden;
  }}
  .det-melhor {{ border-color: var(--yellow); box-shadow: 0 0 0 1px var(--yellow)22; }}
  .det-header {{
    padding: 8px 14px;
    font-size: 11px;
    color: var(--text-dim);
    background: var(--bg3);
    border-bottom: 1px solid var(--border);
  }}
  .det-melhor .det-header {{ color: var(--yellow); }}

  .recorte-row {{ padding: 12px 14px 4px; }}

  .sem-det {{
    padding: 16px;
    color: var(--red);
    font-size: 12px;
  }}

  /* â”€â”€ Variantes â”€â”€ */
  .variantes-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 12px;
    padding: 12px 14px 16px;
  }}
  .var-card {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
  }}
  .var-melhor {{
    border-color: var(--green);
    box-shadow: 0 0 0 1px var(--green)33;
  }}
  .var-label {{
    padding: 6px 10px;
    font-size: 10px;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    background: #ffffff05;
  }}
  .var-melhor .var-label {{ color: var(--green); }}

  .var-card img {{
    width: 100%;
    display: block;
    image-rendering: pixelated;
  }}

  .var-ocr-resultado {{
    padding: 6px 10px;
    font-size: 12px;
    color: var(--text);
    border-top: 1px solid var(--border);
    background: #ffffff04;
  }}

  /* â”€â”€ Candidatos OCR â”€â”€ */
  .candidatos-lista {{
    padding: 6px 10px 10px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}
  .candidato {{
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 10px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }}
  .cand-valida   {{ background: #00ff9d0d; border: 1px solid #00ff9d22; }}
  .cand-invalida {{ background: #ffffff05; border: 1px solid var(--border); }}
  .cand-bruto    {{ color: var(--text-dim); }}
  .cand-norm     {{ color: var(--text); font-weight: 600; }}
  .cand-meta     {{ color: var(--muted); }}
  .sem-ocr       {{ font-size: 10px; color: var(--muted); padding: 4px; }}

  /* â”€â”€ Legenda â”€â”€ */
  .legenda {{
    display: flex; gap: 20px; flex-wrap: wrap;
    margin-bottom: 28px;
    font-size: 11px;
    color: var(--text-dim);
  }}
  .leg-item {{ display: flex; align-items: center; gap: 6px; }}
  .leg-dot  {{
    width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0;
  }}
</style>
</head>
<body>

<h1>ðŸ” Debug OCR Visual</h1>
<p class="subtitle">Gerado em {datetime.now().strftime("%d/%m/%Y Ã s %H:%M:%S")} &nbsp;|&nbsp; {len(registros)} arquivo(s) processado(s)</p>

<div class="legenda">
  <div class="leg-item"><div class="leg-dot" style="background:var(--green)"></div> Leitura correta (= placa esperada)</div>
  <div class="leg-item"><div class="leg-dot" style="background:var(--yellow)"></div> Melhor detecÃ§Ã£o / Melhor variante</div>
  <div class="leg-item"><div class="leg-dot" style="background:var(--red)"></div> Falha / sem leitura</div>
  <div class="leg-item"><div class="leg-dot" style="background:#00e5ff"></div> Leitura parcial</div>
</div>

{cards_html}

</body>
</html>'''

    with open(caminho_saida, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f'\nâœ… RelatÃ³rio salvo em: {caminho_saida}')
    print(f'   Abra no navegador para inspecionar.')


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# FunÃ§Ãµes de extraÃ§Ã£o de dados (PDF e XLSX)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _extrair_de_pdf(caminho_pdf: str) -> tuple[dict, list[bytes]]:
    """Usa a lÃ³gica do ValidadorPDFNovoFormato sem instanciaÃ§Ã£o completa."""
    import fitz
    import pdfplumber

    MIN_AREA = 100_000

    # Dados da passagem
    with pdfplumber.open(caminho_pdf) as pdf:
        tables = pdf.pages[0].extract_tables()

    passagem = {'id': '?', 'placa': '?', 'data': '?', 'hora': '?'}
    if tables:
        for row in tables[0]:
            if row and row[0] and re.match(r'^\d{10,}', str(row[0]).strip()):
                def cel(i):
                    try: return str(row[i]).strip() if row[i] else ''
                    except: return ''
                passagem = {
                    'id': cel(0), 'data': cel(1), 'hora': cel(2),
                    'sp': cel(3), 'km': cel(4), 'praca': cel(5),
                    'pista': cel(6), 'sentido': cel(7), 'municipio': cel(8),
                    'codigo': cel(9), 'placa': cel(10), 'modelo': cel(11),
                    'eixos': cel(12), 'tarifa': cel(13), 'responsavel': cel(14),
                }
                break

    # Imagens
    doc   = fitz.open(caminho_pdf)
    fotos = []
    for img_info in doc[0].get_images(full=True):
        xref = img_info[0]
        d    = doc.extract_image(xref)
        if d['width'] * d['height'] >= MIN_AREA:
            fotos.append(d['image'])
    doc.close()

    return passagem, fotos


def _extrair_de_xlsx(caminho_xlsx: str) -> tuple[dict, list[bytes]]:
    """Usa a lÃ³gica do ValidadorNovoFormato (linha 6 fixa)."""
    import zipfile
    import xml.etree.ElementTree as ET
    from openpyxl import load_workbook

    _NS = {
        'xdr': 'http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing',
        'a':   'http://schemas.openxmlformats.org/drawingml/2006/main',
        'r':   'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
        'rel': 'http://schemas.openxmlformats.org/package/2006/relationships',
    }
    _EXTENSOES_IGNORADAS = {'.emf', '.wmf'}

    wb = load_workbook(caminho_xlsx, data_only=True)
    ws = wb['Fotos'] if 'Fotos' in wb.sheetnames else wb.active

    passagem = {'id': '?', 'placa': '?', 'data': '?', 'hora': '?'}
    for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
        if row_idx == 6:
            def cel(i):
                try: return str(row[i]).strip() if row[i] is not None else ''
                except: return ''
            passagem = {
                'id': cel(1), 'data': cel(2), 'hora': cel(3),
                'sp': cel(4), 'km': cel(5), 'praca': cel(6),
                'pista': cel(7), 'sentido': cel(8), 'municipio': cel(9),
                'codigo': cel(10), 'placa': cel(11), 'modelo': cel(12),
                'eixos': cel(13), 'tarifa': cel(14), 'responsavel': cel(15),
            }
            break

    # Imagens via XML interno
    fotos = []
    with zipfile.ZipFile(caminho_xlsx) as z:
        arquivos = z.namelist()
        rels_sheet = 'xl/worksheets/_rels/sheet1.xml.rels'
        if rels_sheet not in arquivos:
            return passagem, fotos

        sheet_rels   = ET.fromstring(z.read(rels_sheet).decode())
        drawing_path = None
        for rel in sheet_rels.findall('rel:Relationship', _NS):
            if 'drawing' in rel.get('Type', '').lower():
                target = rel.get('Target').lstrip('./')
                drawing_path = 'xl/' + target.lstrip('/')
                break

        if not drawing_path or drawing_path not in arquivos:
            return passagem, fotos

        partes            = drawing_path.rsplit('/', 1)
        drawing_rels_path = partes[0] + '/_rels/' + partes[1] + '.rels'
        if drawing_rels_path not in arquivos:
            return passagem, fotos

        rels_drawing = ET.fromstring(z.read(drawing_rels_path).decode())
        rid_para_arquivo = {
            r.get('Id'): r.get('Target', '').split('/')[-1]
            for r in rels_drawing.findall('rel:Relationship', _NS)
        }

        bytes_midia = {
            a.split('/')[-1]: z.read(a)
            for a in arquivos if a.startswith('xl/media/')
        }

        drawing_root = ET.fromstring(z.read(drawing_path).decode())
        for anchor in drawing_root.findall('xdr:twoCellAnchor', _NS):
            from_elem = anchor.find('xdr:from', _NS)
            if from_elem is None:
                continue
            pic = anchor.find('xdr:pic', _NS)
            if pic is None:
                continue
            blip = pic.find('.//a:blip', _NS)
            if blip is None:
                continue
            rid = blip.get(
                '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed'
            )
            nome = rid_para_arquivo.get(rid, '')
            if not nome:
                continue
            ext = os.path.splitext(nome)[1].lower()
            if ext in _EXTENSOES_IGNORADAS:
                continue
            img_bytes = bytes_midia.get(nome)
            if img_bytes:
                fotos.append(img_bytes)

    return passagem, fotos


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Ponto de entrada
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def main():
    parser = argparse.ArgumentParser(
        description='Gera relatÃ³rio HTML de debug do pipeline YOLO + OCR.'
    )
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument('--pdf',  metavar='CAMINHO', help='Arquivo ou pasta de PDFs')
    grupo.add_argument('--xlsx', metavar='CAMINHO', help='Arquivo ou pasta de XLSXs')

    parser.add_argument('--todos',   action='store_true',
                        help='Processa todos os arquivos da pasta (padrÃ£o: apenas o primeiro)')
    parser.add_argument('--saida',   metavar='HTML', default=None,
                        help='Nome do arquivo HTML de saÃ­da')
    parser.add_argument('--yolo',    metavar='PT',
                        default=PATH_YOLO_PADRAO,
                        help='Caminho para best.pt')
    parser.add_argument('--sem-gpu', action='store_true',
                        help='ForÃ§a CPU')

    args = parser.parse_args()

    # â”€â”€ Resolve lista de arquivos â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ext       = '.pdf' if args.pdf else '.xlsx'
    caminho   = args.pdf or args.xlsx
    extrator  = _extrair_de_pdf if ext == '.pdf' else _extrair_de_xlsx

    if os.path.isdir(caminho):
        arquivos = sorted(glob(os.path.join(caminho, f'*{ext}')))
        if not args.todos:
            arquivos = arquivos[:1]
            print(f'â„¹ï¸  Modo padrÃ£o: processando apenas o primeiro arquivo.')
            print(f'   Use --todos para processar toda a pasta.')
    else:
        arquivos = [caminho]

    if not arquivos:
        print(f'âŒ Nenhum arquivo {ext} encontrado em: {caminho}')
        sys.exit(1)

    print(f'ðŸ“‚ {len(arquivos)} arquivo(s) para processar.')

    # â”€â”€ Inicializa o validador base â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f'\nðŸ¤– Carregando YOLO + EasyOCR...')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from app_excel_spvias import ValidadorExcelPro

    usar_gpu = not args.sem_gpu
    validador = ValidadorExcelPro(
        yolo_weights_path=args.yolo,
        usar_gpu=usar_gpu,
    )
    debug_v = DebugValidador(validador)
    print('âœ… Modelos carregados.\n')

    # â”€â”€ Processa cada arquivo â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    registros = []

    for caminho_arq in arquivos:
        nome = os.path.basename(caminho_arq)
        print(f'ðŸ“„ {nome}')

        passagem, fotos = extrator(caminho_arq)
        print(f'   Placa esperada: {passagem.get("placa","?")}')
        print(f'   {len(fotos)} foto(s) encontrada(s)')

        imgs_debug = []
        MAX_IMAGENS_DEBUG = 500
        for idx, foto_bytes in enumerate(fotos[:MAX_IMAGENS_DEBUG]):
            print(f'   🔍 Imagem {idx+1}/{len(fotos)}...', end=' ', flush=True)
            resultado = debug_v.processar_imagem_com_debug(
                foto_bytes, passagem.get('placa', ''), idx
            )
            imgs_debug.append(resultado)
            print(f'OCR → {resultado["placa_ocr"]}  (score={resultado["melhor_score"]})')

        registros.append({
            'arquivo':   nome,
            'passagem':  passagem,
            'imgs_debug': imgs_debug,
        })

    # â”€â”€ Gera HTML â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    saida   = args.saida or f'debug_ocr_{ts}.html'
    _gerar_html(registros, saida)


if __name__ == '__main__':
    main()


