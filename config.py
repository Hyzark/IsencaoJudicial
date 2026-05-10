from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "detect" / "train1024" / "best.pt"
RESULTS_DIR = BASE_DIR / "results"
DOWNLOADS_DIR = Path.home() / "Downloads"

RESULTS_EVASAO_DIR = RESULTS_DIR / "evasao"
RESULTS_ISENCAO_DIR = RESULTS_DIR / "isencao"
RESULTS_EIXO_DIR = RESULTS_DIR / "eixo"


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
