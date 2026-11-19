"""Shared embedding model, used by both the ingest script and the API."""

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import Settings


def get_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)
