# IsencaoJudicial

Scripts para validar passagens de isencao/evasao usando YOLO + OCR e gerar relatorios em Excel.

## Ambiente

Crie e ative a venv:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Teste a GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Modelo

O modelo principal fica em:

```text
models/detect/train1024/best.pt
```

Esse caminho e as pastas de resultado ficam centralizados em `config.py`.

## Execucao

Os scripts principais estao na raiz:

```powershell
python app_excel_spvias.py
python app_excel_colinas.py
python app_excel_eixovp.py
python app_excel_rota_bandeiras.py
python app_pdf_ecovias.py
python app_pdf_intervias.py
python app_pdf_autoban.py
```

As entradas padrao ficam em subpastas de `Downloads`. Os arquivos gerados sao salvos em `results/`.

## Debug

Para gerar relatorio visual:

```powershell
python debug.py --pdf "C:\caminho\arquivo.pdf"
python debug.py --xlsx "C:\caminho\arquivo.xlsx"
```
