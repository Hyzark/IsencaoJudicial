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

O script main está na raiz:

```powershell
py main.py
```

As entradas padrao ficam em subpastas de `Downloads`. Os arquivos gerados sao salvos em `results/`.

## Debug

Para gerar relatorio visual:

```powershell
Analisar somente um arquivo:
  python debug.py --pdf  caminho/para/arquivo.pdf
  python debug.py --xlsx caminho/para/arquivo.xlsx

Analisar vários arquivos:
  python debug.py --pdf  pasta/com/pdfs/     --todos
  python debug.py --xlsx pasta/com/excels/   --todos
```
