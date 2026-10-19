"""Runtime settings, read from environment variables (12-factor style)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    max_batch_rows: int
    app_password: str
    log_level: str

    @staticmethod
    def from_env() -> "Settings":
        md = Path(os.environ.get("MODEL_DIR", str(ROOT / "models")))
        return Settings(
            model_dir=md if md.is_absolute() else ROOT / md,
            max_batch_rows=int(os.environ.get("MAX_BATCH_ROWS", "5000")),
            app_password=os.environ.get("APP_PASSWORD", ""),
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        )
