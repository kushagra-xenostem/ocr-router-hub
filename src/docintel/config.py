import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
os.environ.setdefault("ONNXTR_CACHE_DIR", str(ROOT / "models" / "onnxtr"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    data_dir: Path
    llm_timeout: float = 300
    api_key: str | None = None
    workers: int = 2
    max_file_mb: int = 50
    max_batch: int = 100
    rate_limit_per_min: int = 120

    @property
    def db_path(self) -> Path:
        return self.data_dir / "docintel.db"

    @property
    def pages_dir(self) -> Path:
        return self.data_dir / "pages"


def get_settings() -> Settings:
    return Settings(
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:8080/v1").rstrip("/"),
        llm_api_key=os.getenv("LLM_API_KEY", "local"),
        llm_model=os.getenv("LLM_MODEL", "qwen3.6-35b-a3b"),
        data_dir=Path(os.getenv("DOCINTEL_DATA", str(ROOT / "data"))),
        llm_timeout=float(os.getenv("LLM_TIMEOUT", "300")),
        api_key=os.getenv("DOCINTEL_API_KEY") or None,
        workers=int(os.getenv("DOCINTEL_WORKERS", "2")),
    )
