"""
Bronze Layer ETL: PDF Parsing with langchain_unstructured.

This script parses PDF files from an input directory using the LangChain
UnstructuredLoader, groups the extracted elements by page, converts them into
clean LangChain Document objects, and persists the results as JSON files in a
Bronze layer output directory.

Output schema (per source PDF JSON file):
    [
        {
            "page_content": "<str>",
            "metadata": {
                "source": "<str>",
                "page": <int>
            }
        },
        ...
    ]

Features:
- Batch processing of multiple PDF files.
- Structured logging for observability.
- Page-level grouping of extracted content.
- Clean LangChain Document handling with consistent metadata.
- Idempotent output: one JSON file per source PDF.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ETLConfig:
    """Configuration for the Bronze Layer PDF ETL process."""

    input_dir: Path = Path("base")
    output_dir: Path = Path("data/bronze/parsed_pdfs")
    log_file: Optional[Path] = Path("logs/bronze_pdf_etl.log")
    batch_size: int = 10
    chunk_strategy: str = "by_page"
    include_metadata: bool = True
    overwrite_existing: bool = False

    def ensure_directories(self) -> None:
        """Create input, output, and log directories if they do not exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def configure_logging(log_file: Optional[Path]) -> logging.Logger:
    """Configure structured logging for the ETL run."""
    logger = logging.getLogger("bronze_pdf_etl")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# PDF Discovery
# ---------------------------------------------------------------------------

def discover_pdfs(input_dir: Path) -> List[Path]:
    """Return a sorted list of PDF files in the input directory."""
    if not input_dir.exists():
        return []
    return sorted(
        path for path in input_dir.rglob("*.pdf")
        if path.is_file() and not path.name.startswith(".")
    )


def batch_paths(paths: List[Path], batch_size: int) -> Iterable[List[Path]]:
    """Yield successive batches of paths of the given size."""
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    for start in range(0, len(paths), batch_size):
        yield paths[start:start + batch_size]


# ---------------------------------------------------------------------------
# LangChain Document Handling
# ---------------------------------------------------------------------------

def load_pdf_documents(pdf_path: Path, logger: logging.Logger) -> List[Document]:
    """
    Load a PDF file using langchain_unstructured and return LangChain Documents.

    The UnstructuredLoader returns LangChain Document objects whose metadata
    typically includes 'page_number' and 'category' fields. Page-level grouping
    is performed post-extraction by the group_documents_by_page function.
    """
    logger.info("Loading PDF with UnstructuredLoader: %s", pdf_path)
    loader = UnstructuredLoader(
        file_path=str(pdf_path),
        strategy="hi_res",
        include_metadata=True,
    )
    documents: List[Document] = loader.load()
    logger.info("Extracted %d raw elements from %s", len(documents), pdf_path.name)
    return documents


def normalize_document(doc: Document, source_file: str) -> Document:
    """
    Normalize a LangChain Document with consistent metadata.

    Ensures required metadata keys exist and values are JSON-serializable.
    """
    metadata: Dict[str, Any] = dict(doc.metadata or {})

    page_number = metadata.get("page_number")
    if page_number is None:
        page_number = metadata.get("page", 1)

    normalized_metadata = {
        "source_file": source_file,
        "source_path": str(metadata.get("source", "")),
        "page_number": int(page_number) if page_number is not None else 1,
        "category": metadata.get("category", "Uncategorized"),
        "element_id": metadata.get("element_id", ""),
        "coordinates": metadata.get("coordinates"),
        "detected_filetype": metadata.get("filetype", "application/pdf"),
        "parsed_at": datetime.now(timezone.utc).isoformat(),
    }

    return Document(page_content=doc.page_content.strip(), metadata=normalized_metadata)


def group_documents_by_page(documents: List[Document]) -> Dict[int, List[Document]]:
    """Group normalized LangChain Documents by their page_number metadata."""
    grouped: Dict[int, List[Document]] = {}
    for doc in documents:
        page = doc.metadata.get("page_number", 1)
        grouped.setdefault(page, []).append(doc)
    return grouped


def build_page_record(
    page_number: int,
    documents: List[Document],
    source_file: str,
) -> Dict[str, Any]:
    """
    Build a JSON-serializable record for a single page.

    Schema:
        {
            "page_content": "<str>",
            "metadata": {
                "source": "<str>",
                "page": <int>
            }
        }
    """
    page_text = "\n\n".join(doc.page_content for doc in documents if doc.page_content)
    return {
        "page_content": page_text,
        "metadata": {
            "source": source_file,
            "page": page_number,
        },
    }


def documents_to_page_records(
    pdf_path: Path,
    documents: List[Document],
) -> List[Dict[str, Any]]:
    """
    Convert a list of LangChain Documents into a flat list of page records.

    Groups elements by page and produces a flat list of objects, where each
    object represents one page and contains exactly 'page_content' (string)
    and 'metadata' (dict with 'source' and 'page' keys).
    """
    source_file = pdf_path.name
    normalized_docs = [normalize_document(doc, source_file) for doc in documents]
    grouped = group_documents_by_page(normalized_docs)

    return [
        build_page_record(page_number, docs, source_file)
        for page_number, docs in sorted(grouped.items())
    ]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_bronze_json(
    source_file: str,
    page_records: List[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Persist a flat list of page records as a JSON file and return its path."""
    output_path = output_dir / f"{Path(source_file).stem}.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(page_records, fh, ensure_ascii=False, indent=2)
    return output_path


def output_exists(pdf_path: Path, output_dir: Path) -> bool:
    """Check whether a Bronze JSON output already exists for the given PDF."""
    return (output_dir / f"{pdf_path.stem}.json").exists()


# ---------------------------------------------------------------------------
# ETL Orchestration
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: Path,
    config: ETLConfig,
    logger: logging.Logger,
) -> Optional[List[Dict[str, Any]]]:
    """Process a single PDF file and return its page records, or None on failure."""
    if not config.overwrite_existing and output_exists(pdf_path, config.output_dir):
        logger.info("Skipping existing output for %s", pdf_path.name)
        return None

    start_time = time.time()
    try:
        documents = load_pdf_documents(pdf_path, logger)
        if not documents:
            logger.warning("No elements extracted from %s", pdf_path.name)
            return None

        page_records = documents_to_page_records(pdf_path, documents)
        output_path = save_bronze_json(pdf_path.name, page_records, config.output_dir)

        elapsed = time.time() - start_time
        logger.info(
            "Saved Bronze record: %s (pages=%d, elements=%d, elapsed=%.2fs)",
            output_path,
            len(page_records),
            len(documents),
            elapsed,
        )
        return page_records

    except Exception as exc:  # noqa: BLE001 - top-level ETL error capture
        logger.exception("Failed to process %s: %s", pdf_path.name, exc)
        return None


def process_batch(
    batch: List[Path],
    config: ETLConfig,
    logger: logging.Logger,
) -> List[List[Dict[str, Any]]]:
    """Process a batch of PDF files and return successful page record lists."""
    logger.info("Processing batch of %d PDF(s)", len(batch))
    results: List[List[Dict[str, Any]]] = []
    for pdf_path in batch:
        page_records = process_pdf(pdf_path, config, logger)
        if page_records is not None:
            results.append(page_records)
    return results


def run_etl(config: ETLConfig) -> List[List[Dict[str, Any]]]:
    """Run the full Bronze Layer PDF ETL pipeline."""
    config.ensure_directories()
    logger = configure_logging(config.log_file)

    logger.info("Starting Bronze Layer PDF ETL")
    logger.info("Input directory: %s", config.input_dir)
    logger.info("Output directory: %s", config.output_dir)
    logger.info("Batch size: %d", config.batch_size)

    pdf_paths = discover_pdfs(config.input_dir)
    if not pdf_paths:
        logger.warning("No PDF files found in %s", config.input_dir)
        return []

    logger.info("Discovered %d PDF file(s)", len(pdf_paths))

    all_records: List[List[Dict[str, Any]]] = []
    for batch_index, batch in enumerate(batch_paths(pdf_paths, config.batch_size), start=1):
        logger.info("--- Batch %d ---", batch_index)
        batch_records = process_batch(batch, config, logger)
        all_records.extend(batch_records)

    logger.info(
        "ETL complete. Processed %d/%d PDF(s) successfully.",
        len(all_records),
        len(pdf_paths),
    )
    return all_records


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the Bronze Layer PDF ETL script."""
    config = ETLConfig(
        input_dir=Path(os.environ.get("ETL_INPUT_DIR", "base")),
        output_dir=Path(os.environ.get("ETL_OUTPUT_DIR", "data/bronze/parsed_pdfs")),
        log_file=Path(os.environ.get("ETL_LOG_FILE", "logs/bronze_pdf_etl.log")),
        batch_size=int(os.environ.get("ETL_BATCH_SIZE", "10")),
        overwrite_existing=os.environ.get("ETL_OVERWRITE", "false").lower() == "true",
    )
    run_etl(config)


if __name__ == "__main__":
    main()