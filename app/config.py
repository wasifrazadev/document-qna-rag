"""Central place for all settings. Values come from environment variables / .env."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _resolve(path_str: str) -> Path:
    """Relative paths are resolved from the project root, so it works from any cwd."""
    path = Path(path_str)
    return path if path.is_absolute() else BASE_DIR / path


@dataclass(frozen=True)
class Settings:
    mistral_model: str
    embedding_model: str
    chroma_dir: Path
    docs_dir: Path
    retriever_k: int
    retriever_fetch_k: int
    retriever_lambda_mult: float
    allowed_origins: list[str]


@lru_cache
def get_settings() -> Settings:
    return Settings(
        mistral_model=os.getenv("MISTRAL_MODEL", "mistral-tiny"),
        # Must be the SAME model for ingestion and querying.
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"
        ),
        chroma_dir=_resolve(os.getenv("CHROMA_DIR", "chroma-db")),
        docs_dir=_resolve(os.getenv("DOCS_DIR", "docs")),
        retriever_k=int(os.getenv("RETRIEVER_K", "4")),
        retriever_fetch_k=int(os.getenv("RETRIEVER_FETCH_K", "10")),
        retriever_lambda_mult=float(os.getenv("RETRIEVER_LAMBDA_MULT", "0.5")),
        allowed_origins=[
            o.strip()
            for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")
            if o.strip()
        ],
    )
