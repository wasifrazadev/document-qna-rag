"""
One-time (or whenever your PDFs change) script that reads PDFs, splits them
into chunks, embeds them and stores them in the Chroma vector database.

Usage (from the project root):
    python -m scripts.ingest                  # ingest every PDF in docs/
    python -m scripts.ingest docs/pp.pdf      # ingest a single PDF
    python -m scripts.ingest --reset          # wipe the DB first, then ingest
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.embeddings import get_embeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ingest")


def collect_pdfs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.pdf"))
    return []


def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Create embeddings from PDFs.")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=settings.docs_dir,
        help="A PDF file or a folder of PDFs (default: docs/)",
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=60)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing vector store before ingesting",
    )
    args = parser.parse_args()

    pdfs = collect_pdfs(args.path)
    if not pdfs:
        logger.error("No PDF files found at '%s'.", args.path)
        sys.exit(1)

    if args.reset and settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
        logger.info("Removed existing vector store.")
    elif settings.chroma_dir.exists():
        logger.warning(
            "Vector store already exists. Re-ingesting the same PDF will create "
            "duplicates. Use --reset to start fresh."
        )

    docs = []
    for pdf in pdfs:
        logger.info("Loading %s", pdf)
        docs.extend(PyPDFLoader(str(pdf)).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split %d pages into %d chunks.", len(docs), len(chunks))

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(settings),
        persist_directory=str(settings.chroma_dir),
    )
    logger.info("Done. Vector store saved to '%s'.", settings.chroma_dir)


if __name__ == "__main__":
    main()
