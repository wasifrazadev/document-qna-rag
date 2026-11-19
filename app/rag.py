"""RAG logic: retrieve relevant chunks, build the prompt, call the LLM."""

import logging
import os

from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI

from app.config import Settings
from app.embeddings import get_embeddings
from app.prompts import RAG_PROMPT

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, settings: Settings):
        if not os.getenv("MISTRAL_API_KEY"):
            raise RuntimeError(
                "MISTRAL_API_KEY is not set. Copy .env.example to .env and add your key."
            )

        if not settings.chroma_dir.exists():
            raise RuntimeError(
                f"Vector store not found at '{settings.chroma_dir}'. "
                "Run `python -m scripts.ingest` first to create it."
            )

        logger.info("Loading embedding model: %s", settings.embedding_model)
        vstore = Chroma(
            persist_directory=str(settings.chroma_dir),
            embedding_function=get_embeddings(settings),
        )

        self.retriever = vstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.retriever_k,
                "fetch_k": settings.retriever_fetch_k,
                "lambda_mult": settings.retriever_lambda_mult,
            },
        )
        self.llm = ChatMistralAI(model=settings.mistral_model)
        self.prompt = RAG_PROMPT

    def ask(self, question: str) -> tuple[str, list[str]]:
        docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        final_prompt = self.prompt.invoke(
            {"context": context, "question": question}
        )
        response = self.llm.invoke(final_prompt)

        return response.content, [doc.page_content for doc in docs]
