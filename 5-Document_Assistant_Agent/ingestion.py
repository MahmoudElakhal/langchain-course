"""Crawl LangChain documentation and store its embeddings in local Chroma."""

import asyncio
import hashlib
import os
from pathlib import Path
from typing import List

import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilyCrawl
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logger import Colors, log_header, log_info, log_success


# Resolve paths relative to this file, NOT to the terminal's working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# Keep the existing certificate configuration for Tavily/OpenAI HTTPS calls.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

CHROMA_DIRECTORY = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "langchain-docs"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_max_seconds=10,
)

chroma = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(CHROMA_DIRECTORY),
    embedding_function=embeddings,
)

tavily_crawl = TavilyCrawl()


def _document_ids(documents: List[Document]) -> List[str]:
    """Stable IDs prevent duplicate records when ingesting the same chunks again."""
    return [
        hashlib.sha256(
            f"{index}\0{doc.metadata.get('source', '')}\0{doc.page_content}".encode("utf-8")
        ).hexdigest()
        for index, doc in enumerate(documents)
    ]


async def index_documents_async(documents: List[Document], batch_size: int = 50) -> None:
    """Embed and add documents in modest, sequential batches to local Chroma."""
    if not documents:
        raise ValueError("No documents to index. Check the Tavily crawl results.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    log_header("VECTOR STORAGE PHASE")
    log_info(f"Indexing {len(documents)} chunks into {CHROMA_DIRECTORY}", Colors.DARKCYAN)

    ids = _document_ids(documents)
    number_of_batches = (len(documents) + batch_size - 1) // batch_size

    # Sequential writes avoid unnecessary contention on the local database.
    for batch_number, start in enumerate(range(0, len(documents), batch_size), start=1):
        batch = documents[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]
        await chroma.aadd_documents(documents=batch, ids=batch_ids)
        log_success(
            f"Indexed batch {batch_number}/{number_of_batches} "
            f"({len(batch)} chunks)"
        )

    log_success("All document batches indexed successfully")


async def main() -> None:
    """Crawl documentation, chunk it, and index it into persistent Chroma."""
    log_header("DOCUMENTATION INGESTION PIPELINE")
    log_info("Crawling LangChain documentation", Colors.PURPLE)

    response = await tavily_crawl.ainvoke(
        {
            "url": "https://docs.langchain.com/oss/python/langchain/overview",
            "max_depth": 2,
            "extract_depth": "advanced",
        }
    )

    all_docs: List[Document] = []
    for result in response.get("results", []):
        source = result.get("url", "")
        content = result.get("raw_content") or ""
        if not source or not content.strip():
            continue
        all_docs.append(Document(page_content=content, metadata={"source": source}))
        log_info(f"Crawled: {source}")

    if not all_docs:
        raise RuntimeError("Tavily returned no documentation content. Nothing was indexed.")

    log_header("DOCUMENT CHUNKING PHASE")
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)
    log_info(f"Created {len(split_docs)} chunks from {len(all_docs)} pages", Colors.YELLOW)

    await index_documents_async(split_docs, batch_size=50)

    log_header("PIPELINE COMPLETE")
    log_success("Local Chroma documentation index updated successfully")
    log_info(f"Database: {CHROMA_DIRECTORY}")
    log_info(f"Collection: {COLLECTION_NAME}")
    log_info(f"Pages crawled: {len(all_docs)}")
    log_info(f"Chunks indexed: {len(split_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
