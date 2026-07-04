"""Silver Layer: Chunking and Embedding pipeline.

Production-grade module that transforms the flat JSON produced by the Bronze
layer into a searchable ChromaDB vector store.

Pipeline steps:
    1. Read the flat JSON file (list of dicts with `page_content` and `metadata`).
    2. Convert each item into a LangChain ``Document``.
    3. Apply a modular chunking strategy selected through a Factory pattern
       (currently supporting ``RecursiveCharacterTextSplitter`` and
       ``TokenTextSplitter``).
    4. Ingest the resulting chunks into a ChromaDB vector store using
       ``OpenAIEmbeddings`` from ``langchain_openai``, with batched insertion
       to avoid OpenAI API rate limits.

The module is designed to be imported as a library or executed as a CLI
script.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Type aliases
# --------------------------------------------------------------------------- #

ChunkStrategyName = str  # e.g. "recursive", "token"


# --------------------------------------------------------------------------- #
# JSON loading and Document conversion
# --------------------------------------------------------------------------- #


def load_items_from_json(json_path: str | Path) -> List[Dict[str, Any]]:
    """Load the flat list of items produced by the Bronze layer.

    Each item is expected to be a dict containing at least a ``page_content``
    key and optionally a ``metadata`` key.

    Args:
        json_path: Path to the JSON file produced by the Bronze layer.

    Returns:
        A list of dictionaries.

    Raises:
        FileNotFoundError: If ``json_path`` does not exist.
        ValueError: If the JSON content is not a list.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list of items, received: {type(data).__name__}"
        )

    logger.info("Loaded %d items from %s", len(data), json_path)
    return data


def items_to_documents(items: List[Dict[str, Any]]) -> List[Document]:
    """Convert Bronze-layer items into LangChain ``Document`` objects.

    Each item must contain a ``page_content`` key. A ``metadata`` key, if
    present and a dict, is preserved. Items without meaningful content are
    skipped with a warning.

    Args:
        items: List of dicts with ``page_content`` and optional ``metadata``.

    Returns:
        List of LangChain ``Document`` instances.
    """
    documents: List[Document] = []

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            logger.warning("Item #%d is not a dict and was skipped", idx)
            continue

        content = item.get("page_content")
        if content is None or not str(content).strip():
            logger.warning("Item #%d has empty page_content and was skipped", idx)
            continue

        metadata = item.get("metadata")
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            logger.warning(
                "Item #%d has non-dict metadata (%s); coercing to empty dict",
                idx,
                type(metadata).__name__,
            )
            metadata = {}

        documents.append(
            Document(page_content=str(content), metadata=dict(metadata))
        )

    logger.info("Converted %d items into LangChain Document(s)", len(documents))
    return documents


# --------------------------------------------------------------------------- #
# Chunking: Factory pattern
# --------------------------------------------------------------------------- #


class TextSplitter(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def split(self, documents: List[Document]) -> List[Document]:
        """Split a list of documents into chunks.

        Args:
            documents: List of LangChain ``Document`` objects.

        Returns:
            List of chunked ``Document`` objects.
        """
        raise NotImplementedError


@dataclass
class RecursiveCharacterSplitter(TextSplitter):
    """Chunking strategy based on ``RecursiveCharacterTextSplitter``."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: Optional[List[str]] = None

    def split(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators if self.separators is not None else ["\n\n", "\n", " ", ""],
        )
        return splitter.split_documents(documents)


@dataclass
class TokenSplitter(TextSplitter):
    """Chunking strategy based on ``TokenTextSplitter``."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    encoding_name: str = "cl100k_base"
    model_name: Optional[str] = None

    def split(self, documents: List[Document]) -> List[Document]:
        kwargs: Dict[str, Any] = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "encoding_name": self.encoding_name,
        }
        if self.model_name is not None:
            kwargs["model_name"] = self.model_name
        splitter = TokenTextSplitter(**kwargs)
        return splitter.split_documents(documents)


class SplitterFactory:
    """Factory that produces ``TextSplitter`` instances by name."""

    _registry: Dict[str, Callable[..., TextSplitter]] = {}

    @classmethod
    def register(
        cls, name: str, builder: Callable[..., TextSplitter]
    ) -> None:
        """Register a new splitter builder under ``name``."""
        cls._registry[name] = builder

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> TextSplitter:
        """Create a splitter instance by registered name.

        Args:
            name: Registered strategy name (e.g. ``"recursive"``, ``"token"``).
            **kwargs: Strategy-specific parameters.

        Returns:
            A ``TextSplitter`` instance.

        Raises:
            ValueError: If ``name`` is not registered.
        """
        builder = cls._registry.get(name)
        if builder is None:
            raise ValueError(
                f"Unknown chunking strategy: '{name}'. "
                f"Available: {sorted(cls._registry.keys())}"
            )
        return builder(**kwargs)

    @classmethod
    def available_strategies(cls) -> List[str]:
        """Return the list of registered strategy names."""
        return sorted(cls._registry.keys())


# Register built-in strategies.
SplitterFactory.register("recursive", RecursiveCharacterSplitter)
SplitterFactory.register("token", TokenSplitter)


def chunk_documents(
    documents: List[Document],
    strategy: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **strategy_kwargs: Any,
) -> List[Document]:
    """Apply a chunking strategy to a list of documents.

    Args:
        documents: List of LangChain ``Document`` objects.
        strategy: Name of the registered chunking strategy.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap between consecutive chunks.
        **strategy_kwargs: Additional strategy-specific parameters.

    Returns:
        List of chunked ``Document`` objects.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = SplitterFactory.create(
        strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **strategy_kwargs,
    )

    chunks = splitter.split(documents)
    logger.info(
        "Chunking completed: %d documents -> %d chunks (strategy=%s, "
        "chunk_size=%d, chunk_overlap=%d)",
        len(documents),
        len(chunks),
        strategy,
        chunk_size,
        chunk_overlap,
    )
    return chunks


# --------------------------------------------------------------------------- #
# Embeddings and vector store ingestion
# --------------------------------------------------------------------------- #


def build_embeddings(
    model: str = "text-embedding-3-small",
    openai_api_key: Optional[str] = None,
    **openai_kwargs: Any,
) -> OpenAIEmbeddings:
    """Build an ``OpenAIEmbeddings`` instance.

    The OpenAI API key is resolved from the explicit argument or the
    ``OPENAI_API_KEY`` environment variable.

    Args:
        model: OpenAI embedding model name.
        openai_api_key: Optional API key. Falls back to env var.
        **openai_kwargs: Extra kwargs forwarded to ``OpenAIEmbeddings``.

    Returns:
        An ``OpenAIEmbeddings`` instance.
    """
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not provided. Set OPENAI_API_KEY env var or pass "
            "openai_api_key explicitly."
        )

    logger.info("Initializing OpenAIEmbeddings (model=%s)", model)
    return OpenAIEmbeddings(model=model, api_key=SecretStr(api_key), **openai_kwargs)


def _batched(iterable: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    """Yield successive batches of size ``batch_size`` from ``iterable``."""
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def ingest_into_chroma(
    chunks: List[Document],
    embeddings: OpenAIEmbeddings,
    collection_name: str,
    persist_directory: str | Path = "./chroma_db",
    batch_size: int = 100,
    batch_delay_seconds: float = 0.0,
) -> Chroma:
    """Ingest chunks into a ChromaDB vector store in batches.

    Batching mitigates OpenAI API rate limits by limiting the number of texts
    embedded and inserted per request. An optional delay between batches
    provides additional throttling.

    Args:
        chunks: List of chunked ``Document`` objects.
        embeddings: An ``OpenAIEmbeddings`` instance.
        collection_name: Name of the ChromaDB collection.
        persist_directory: Directory where ChromaDB will persist data.
        batch_size: Number of chunks per insertion batch.
        batch_delay_seconds: Seconds to wait between batches.

    Returns:
        The populated ``Chroma`` vector store instance.
    """
    if not chunks:
        raise ValueError("No chunks provided for ingestion")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Ingesting %d chunks into collection '%s' at %s (batch_size=%d, "
        "batch_delay=%.2fs)",
        len(chunks),
        collection_name,
        persist_directory,
        batch_size,
        batch_delay_seconds,
    )

    vectorstore: Optional[Chroma] = None
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for batch_idx, batch in enumerate(_batched(chunks, batch_size), start=1):
        logger.info(
            "Inserting batch %d/%d (%d chunks) into collection '%s'",
            batch_idx,
            total_batches,
            len(batch),
            collection_name,
        )

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=str(persist_directory),
            )
        else:
            vectorstore.add_documents(documents=batch)

        if batch_delay_seconds > 0 and batch_idx < total_batches:
            time.sleep(batch_delay_seconds)

    if vectorstore is None:  # pragma: no cover - defensive guard
        raise RuntimeError("Failed to initialize vector store")

    logger.info("Ingestion completed for collection '%s'", collection_name)
    return vectorstore


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def get_collection_name(strategy: str, base_name: str = "rag_docs") -> str:
    """Generate a dynamic collection name based on the chunking strategy.

    Args:
        strategy: Chunking strategy name.
        base_name: Base collection name.

    Returns:
        Collection name in the form ``f"{base_name}_{strategy}"``.
    """
    return f"{base_name}_{strategy}"


def run_pipeline(
    json_path: str | Path,
    strategy: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    persist_directory: str | Path = "./chroma_db",
    embedding_model: str = "text-embedding-3-small",
    openai_api_key: Optional[str] = None,
    base_collection_name: str = "rag_docs",
    batch_size: int = 100,
    batch_delay_seconds: float = 0.0,
    strategy_kwargs: Optional[Dict[str, Any]] = None,
) -> Chroma:
    """Execute the full Silver-layer pipeline.

    Reads the Bronze-layer JSON, converts items into LangChain Documents,
    applies a configurable chunking strategy, and ingests the chunks into a
    ChromaDB vector store using OpenAI embeddings with batched insertion.

    Args:
        json_path: Path to the Bronze-layer JSON file.
        strategy: Chunking strategy name (e.g. ``"recursive"``, ``"token"``).
        chunk_size: Maximum chunk size.
        chunk_overlap: Overlap between consecutive chunks.
        persist_directory: ChromaDB persistence directory.
        embedding_model: OpenAI embedding model name.
        openai_api_key: Optional OpenAI API key.
        base_collection_name: Base collection name (suffixed by strategy).
        batch_size: Number of chunks per ingestion batch.
        batch_delay_seconds: Delay between batches in seconds.
        strategy_kwargs: Additional strategy-specific parameters.

    Returns:
        The populated ``Chroma`` vector store instance.
    """
    strategy_kwargs = strategy_kwargs or {}

    items = load_items_from_json(json_path)
    documents = items_to_documents(items)
    chunks = chunk_documents(
        documents,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **strategy_kwargs,
    )

    embeddings = build_embeddings(
        model=embedding_model,
        openai_api_key=openai_api_key,
    )

    collection_name = get_collection_name(strategy, base_name=base_collection_name)
    vectorstore = ingest_into_chroma(
        chunks=chunks,
        embeddings=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
        batch_size=batch_size,
        batch_delay_seconds=batch_delay_seconds,
    )
    return vectorstore


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Silver Layer: chunk and embed Bronze-layer JSON into a ChromaDB "
            "vector store using OpenAI embeddings."
        )
    )
    parser.add_argument("json_path", type=str, help="Path to the Bronze-layer JSON file")
    parser.add_argument(
        "--strategy",
        type=str,
        default="recursive",
        choices=SplitterFactory.available_strategies(),
        help="Chunking strategy",
    )
    parser.add_argument("--chunk-size", type=int, default=1000, help="Maximum chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    parser.add_argument(
        "--persist-directory",
        type=str,
        default="./chroma_db",
        help="ChromaDB persistence directory",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model name",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--base-collection-name",
        type=str,
        default="rag_docs",
        help="Base collection name (suffixed by strategy)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of chunks per vector DB insertion batch",
    )
    parser.add_argument(
        "--batch-delay-seconds",
        type=float,
        default=0.0,
        help="Delay (seconds) between insertion batches to avoid rate limits",
    )
    parser.add_argument(
        "--token-encoding",
        type=str,
        default="cl100k_base",
        help="Token encoding for the 'token' strategy",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    # Load environment variables from a .env file (if present) before any
    # configuration or pipeline execution so that secrets such as
    # OPENAI_API_KEY are available to downstream components.
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    strategy_kwargs: Dict[str, Any] = {}
    if args.strategy == "token":
        strategy_kwargs["encoding_name"] = args.token_encoding

    run_pipeline(
        json_path=args.json_path,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        persist_directory=args.persist_directory,
        embedding_model=args.embedding_model,
        openai_api_key=args.openai_api_key,
        base_collection_name=args.base_collection_name,
        batch_size=args.batch_size,
        batch_delay_seconds=args.batch_delay_seconds,
        strategy_kwargs=strategy_kwargs,
    )


if __name__ == "__main__":
    main()