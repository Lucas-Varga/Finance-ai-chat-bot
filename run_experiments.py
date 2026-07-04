#!/usr/bin/env python3
"""
run_experiments.py — Batch experiment runner for RAG strategy tuning.

Tests the full matrix of chunking x retrieval strategies against the test
dataset, scores each combination via LLM-as-Judge, and optionally auto-updates
the config with the champion.

Usage:
    python run_experiments.py --mode matrix        # test chunk x retrieval matrix
    python run_experiments.py --mode auto-tune     # matrix + auto-update config
    python run_experiments.py --guide              # print parameter guide
    python run_experiments.py --ingest-only        # create all collections

Inspired by the video "Como CRIAR um Agente de IA RAG PERFEITO (5 passos)" —
the author tests multiple chunking+retrieval combinations, finds the champion,
and auto-tunes the config. This script replicates that exact workflow.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv

from gold_rag_agent import (
    AgentConfig,
    GoldRAGAgent,
    RetrieverStrategy,
)

from evaluator_llm_judge import RAGEvaluator

import chromadb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ExperimentRunner")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ChunkingConfig:
    """Defines a chunking strategy for experiments."""
    name: str
    strategy: str
    chunk_size: int
    chunk_overlap: int


@dataclass
class ExperimentKey:
    """Uniquely identifies one cell in the chunk x retrieval matrix."""
    chunk_name: str
    retrieval_strategy: str
    collection_name: str


@dataclass
class ExperimentResult:
    """Result of a single chunk x retrieval combination."""
    key: ExperimentKey
    total_questions: int
    passed: int
    failed: int
    pass_rate: float
    avg_context_relevance: float
    avg_answer_correctness: float
    avg_score: float


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Orchestrates the chunk x retrieval matrix experiment.

    Flow:
      1. Load test dataset
      2. Load chunking + retrieval strategies from YAML
      3. Ensure all collections exist (ingest if needed)
      4. Evaluate every (chunk, retrieval) combination
      5. Find champion
      6. Optionally update rag_config.yaml
    """

    _PARAM_GUIDE = {
        "chunking.strategy": {
            "recursive": "RecursiveCharacterTextSplitter. Splits on newlines, then spaces, then chars. Good general-purpose.",
            "token": "TokenTextSplitter. Splits on token boundaries. Good for precise token budget.",
        },
        "chunk_size": "Characters/tokens per chunk (256-2000). Smaller = more precise chunks but less context. Default: 1000.",
        "chunk_overlap": "Overlap between chunks (10-20% of chunk_size). Maintains context at boundaries. Default: 200.",
        "retrieval.strategy": {
            "standard": "Dense cosine similarity. Fast baseline.",
            "mmr": "Maximal Marginal Relevance. Balances relevance & diversity.",
            "hyde": "Hypothetical Document Embeddings. Generates hypothetical answer before search.",
        },
        "top_k": "Number of documents retrieved (1-10). Default: 4.",
        "temperature": "LLM creativity (0-1). 0 = deterministic. Default: 0.1.",
    }

    SCORE_THRESHOLD_PASS = 3

    def __init__(
        self,
        config_path: str = "rag_config.yaml",
        test_csv: str = "test_dataset.csv",
        bronze_json: Optional[str] = None,
    ):
        self.config_path = config_path
        self.test_csv = test_csv
        self.bronze_json = bronze_json
        self.questions: List[Dict[str, str]] = []
        self.results: List[ExperimentResult] = []
        self.chunking_configs: List[ChunkingConfig] = []
        self.retrieval_strategies: List[str] = []
        self.base_collection: str = "finance_docs"
        self.chroma_path: str = "./chroma_db"

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def load_experiment_config(self) -> None:
        """Load chunking + retrieval strategies from rag_config.yaml."""
        cfg = self._load_yaml()

        # Base collection name
        silver = cfg.get("silver_layer", {})
        vector_db = silver.get("vector_db", {})
        self.base_collection = vector_db.get("base_collection_name", "finance_docs")
        self.chroma_path = vector_db.get("path", "./chroma_db")

        # Experiment strategies
        exp = cfg.get("experiment", {})

        raw_chunking = exp.get("chunking_strategies", [])
        self.chunking_configs = [
            ChunkingConfig(
                name=c["name"],
                strategy=c.get("strategy", "recursive"),
                chunk_size=c.get("chunk_size", 1000),
                chunk_overlap=c.get("chunk_overlap", 200),
            )
            for c in raw_chunking
        ]

        self.retrieval_strategies = exp.get("retrieval_strategies", ["standard"])

        logger.info(
            "Experiment: %d chunking x %d retrieval = %d combinations",
            len(self.chunking_configs),
            len(self.retrieval_strategies),
            len(self.chunking_configs) * len(self.retrieval_strategies),
        )

    def _collection_name(self, chunk_name: str) -> str:
        return f"{self.base_collection}_{chunk_name}"

    def _load_yaml(self) -> Dict[str, Any]:
        with open(self.config_path) as f:
            return yaml.safe_load(f) or {}

    def _save_yaml(self, cfg: Dict[str, Any]) -> None:
        with open(self.config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        logger.info("Config saved to %s", self.config_path)

    def load_test_data(self) -> None:
        if not os.path.exists(self.test_csv):
            raise FileNotFoundError(f"Test dataset not found: {self.test_csv}")
        with open(self.test_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.questions = list(reader)
        logger.info("Loaded %d test questions", len(self.questions))

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def list_collections(self) -> List[str]:
        """Return list of existing collection names in ChromaDB."""
        client = chromadb.PersistentClient(path=self.chroma_path)
        return [c.name for c in client.list_collections()]

    def ensure_collections(self) -> None:
        """
        For each chunking config, run silver pipeline if collection
        does not exist yet.
        """
        existing = self.list_collections()
        json_path = self.bronze_json

        if not json_path:
            # Auto-detect: use first JSON in data/bronze/parsed_pdfs/
            bronze_dir = "data/bronze/parsed_pdfs"
            json_files = sorted(
                os.path.join(bronze_dir, f)
                for f in os.listdir(bronze_dir)
                if f.endswith(".json")
            ) if os.path.isdir(bronze_dir) else []
            if json_files:
                json_path = json_files[0]
                logger.info("Auto-detected bronze JSON: %s", json_path)

        if not json_path or not os.path.exists(json_path):
            logger.warning(
                "No bronze JSON found. Run 'make bronze' first. "
                "Skipping ingestion, testing only existing collections."
            )
            return

        for chunk_cfg in self.chunking_configs:
            col_name = self._collection_name(chunk_cfg.name)
            if col_name in existing:
                count = self._collection_count(col_name)
                if count > 0:
                    logger.info(
                        "Collection '%s' exists (%d docs), skipping ingestion",
                        col_name, count,
                    )
                    continue
                logger.info("Collection '%s' exists but empty, re-ingesting", col_name)

            logger.info(
                "Ingesting collection '%s' (strategy=%s, size=%d, overlap=%d)...",
                col_name, chunk_cfg.strategy, chunk_cfg.chunk_size, chunk_cfg.chunk_overlap,
            )
            self._run_silver_pipeline(
                json_path=json_path,
                collection_name=col_name,
                strategy=chunk_cfg.strategy,
                chunk_size=chunk_cfg.chunk_size,
                chunk_overlap=chunk_cfg.chunk_overlap,
            )

    def _collection_count(self, name: str) -> int:
        client = chromadb.PersistentClient(path=self.chroma_path)
        try:
            return client.get_collection(name).count()
        except Exception:
            return 0

    def _run_silver_pipeline(
        self,
        json_path: str,
        collection_name: str,
        strategy: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        """
        Execute silver_chunk_embed.py pipeline for a specific chunking config.
        We import and call directly to avoid subprocess overhead.
        """
        from silver_chunk_embed import (
            load_items_from_json,
            items_to_documents,
            chunk_documents,
            build_embeddings,
            ingest_into_chroma,
        )

        logger.info("Loading items from %s...", json_path)
        items = load_items_from_json(json_path)
        docs = items_to_documents(items)

        logger.info(
            "Chunking: strategy=%s, size=%d, overlap=%d",
            strategy, chunk_size, chunk_overlap,
        )
        chunks = chunk_documents(
            docs,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        embeddings = build_embeddings()
        ingest_into_chroma(
            chunks=chunks,
            embeddings=embeddings,
            collection_name=collection_name,
            persist_directory=self.chroma_path,
        )
        logger.info("Collection '%s' created with %d chunks", collection_name, len(chunks))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _build_evaluator_for_key(self, key: ExperimentKey) -> RAGEvaluator:
        """
        Build an evaluator that uses a specific collection + retrieval strategy.
        We write a temp YAML config for RAGEvaluator to read.
        """
        cfg = self._load_yaml()

        # Ensure silver_layer.vector_db.base_collection_name matches
        if "silver_layer" not in cfg:
            cfg["silver_layer"] = {}
        if "vector_db" not in cfg["silver_layer"]:
            cfg["silver_layer"]["vector_db"] = {}
        cfg["silver_layer"]["vector_db"]["base_collection_name"] = self.base_collection

        # Override chunking config so collection name derivation works
        # Actually, we need to match the EXACT collection name.
        # RAGEvaluator derives: f"{base_collection}_{chunk_strategy}"
        # Our collection name is: f"{base_collection}_{chunk_name}"
        # So we set chunking.strategy = key.chunk_name so it matches.
        if "chunking" not in cfg["silver_layer"]:
            cfg["silver_layer"]["chunking"] = {}
        cfg["silver_layer"]["chunking"]["strategy"] = key.chunk_name

        # Override gold_layer.retrieval
        if "gold_layer" not in cfg:
            cfg["gold_layer"] = {}
        if "retrieval" not in cfg["gold_layer"]:
            cfg["gold_layer"]["retrieval"] = {}
        cfg["gold_layer"]["retrieval"]["strategy"] = key.retrieval_strategy

        # Write temp config
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
            temp_path = f.name

        return temp_path

    def _evaluate_combination(self, key: ExperimentKey) -> ExperimentResult:
        """Evaluate one (chunk, retrieval) combination."""
        logger.info("=" * 60)
        logger.info("Evaluating: chunk=%s | retrieval=%s", key.chunk_name, key.retrieval_strategy)
        logger.info("Collection: %s", key.collection_name)
        logger.info("=" * 60)

        temp_path = self._build_evaluator_for_key(key)
        try:
            evaluator = RAGEvaluator(config_path=temp_path)
            results_df = evaluator.run_evaluation(
                csv_path=self.test_csv,
                output_path=f"evaluation_{key.chunk_name}_{key.retrieval_strategy}.csv",
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        total = len(results_df)
        passed = int(results_df["overall_pass"].sum())
        failed = total - passed
        avg_relevance = results_df["context_relevance_score"].mean()
        avg_correctness = results_df["answer_correctness_score"].mean()
        avg_score = (avg_relevance + avg_correctness) / 2
        pass_rate = passed / total if total > 0 else 0.0

        logger.info(
            "Result: score=%.2f, relevance=%.2f, correctness=%.2f, pass_rate=%.0f%%",
            avg_score, avg_relevance, avg_correctness, pass_rate * 100,
        )

        return ExperimentResult(
            key=key,
            total_questions=total,
            passed=passed,
            failed=failed,
            pass_rate=pass_rate,
            avg_context_relevance=avg_relevance,
            avg_answer_correctness=avg_correctness,
            avg_score=avg_score,
        )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_matrix(self) -> List[ExperimentResult]:
        """Run the full chunk x retrieval matrix."""
        self.load_test_data()
        self.load_experiment_config()
        self.ensure_collections()

        self.results = []

        for chunk_cfg in self.chunking_configs:
            col_name = self._collection_name(chunk_cfg.name)
            for retr_strat in self.retrieval_strategies:
                key = ExperimentKey(
                    chunk_name=chunk_cfg.name,
                    retrieval_strategy=retr_strat,
                    collection_name=col_name,
                )
                try:
                    result = self._evaluate_combination(key)
                    self.results.append(result)
                except Exception as e:
                    logger.error(
                        "Failed: chunk=%s, retrieval=%s: %s",
                        chunk_cfg.name, retr_strat, e,
                    )

        return self.results

    def rank_results(self) -> List[ExperimentResult]:
        return sorted(self.results, key=lambda r: r.avg_score, reverse=True)

    def find_champion(self) -> Tuple[Optional[ExperimentResult], List[ExperimentResult]]:
        ranked = self.rank_results()
        return (ranked[0] if ranked else None), ranked

    def auto_tune(self) -> Optional[ExperimentResult]:
        """
        Full auto-tune:
        1. Run chunk x retrieval matrix
        2. Find champion
        3. Print report
        4. Update rag_config.yaml with champion parameters
        """
        logger.info("=" * 60)
        logger.info("AUTO-TUNE MODE — Testing chunk x retrieval matrix")
        logger.info("=" * 60)

        self.run_matrix()
        champion, ranked = self.find_champion()

        self._print_report(champion, ranked)

        if champion:
            self._update_config_with_champion(champion)

        return champion

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def _print_report(
        self, champion: Optional[ExperimentResult], ranked: List[ExperimentResult]
    ) -> None:
        print()
        print("=" * 80)
        print("  EXPERIMENT REPORT — Chunk x Retrieval Matrix")
        print("=" * 80)
        print()

        if not ranked:
            print("  No results.")
            return

        print(
            f"  {'Chunking':<22} {'Retrieval':<12} "
            f"{'Score':>6} {'Relev.':>6} {'Corr.':>6} {'Pass%':>6}"
        )
        print(
            f"  {'-'*22} {'-'*12} "
            f"{'-'*6} {'-'*6} {'-'*6} {'-'*6}"
        )
        for r in ranked:
            marker = " ◄ CHAMPION" if r is champion else ""
            print(
                f"  {r.key.chunk_name:<22} {r.key.retrieval_strategy:<12} "
                f"{r.avg_score:>5.2f} "
                f"{r.avg_context_relevance:>5.2f} "
                f"{r.avg_answer_correctness:>5.2f} "
                f"{r.pass_rate:>5.0%}{marker}"
            )

        print()
        if champion:
            print(f"  🏆 Champion: chunk='{champion.key.chunk_name}' + retrieval='{champion.key.retrieval_strategy}'")
            print(f"     Score: {champion.avg_score:.2f}/5 | Pass Rate: {champion.pass_rate:.0%}")
            print(f"     Collection: {champion.key.collection_name}")
        print("=" * 80)
        print()

    def _update_config_with_champion(self, champion: ExperimentResult) -> None:
        """Update rag_config.yaml with champion chunking + retrieval, and
        update silver/gold layer configs to match."""
        cfg = self._load_yaml()

        # Find the corresponding chunking config
        chunk_cfg = next(
            (c for c in self.chunking_configs if c.name == champion.key.chunk_name),
            None,
        )

        # Update silver layer
        if "silver_layer" not in cfg:
            cfg["silver_layer"] = {}
        if "chunking" not in cfg["silver_layer"]:
            cfg["silver_layer"]["chunking"] = {}
        if chunk_cfg:
            cfg["silver_layer"]["chunking"]["strategy"] = chunk_cfg.strategy
            cfg["silver_layer"]["chunking"]["chunk_size"] = chunk_cfg.chunk_size
            cfg["silver_layer"]["chunking"]["chunk_overlap"] = chunk_cfg.chunk_overlap

        # Update gold layer
        if "gold_layer" not in cfg:
            cfg["gold_layer"] = {}
        if "retrieval" not in cfg["gold_layer"]:
            cfg["gold_layer"]["retrieval"] = {}
        cfg["gold_layer"]["retrieval"]["strategy"] = champion.key.retrieval_strategy

        cfg["experiment_name"] = (
            f"champion_{champion.key.chunk_name}_{champion.key.retrieval_strategy}"
        )

        self._save_yaml(cfg)
        logger.info(
            "Config updated: chunk=%s (size=%s, overlap=%s), retrieval=%s",
            champion.key.chunk_name,
            chunk_cfg.chunk_size if chunk_cfg else "?",
            chunk_cfg.chunk_overlap if chunk_cfg else "?",
            champion.key.retrieval_strategy,
        )

    # ------------------------------------------------------------------
    # Documentation
    # ------------------------------------------------------------------

    @staticmethod
    def param_guide() -> str:
        lines = [
            "=== RAG System Parameter Guide ===",
            "",
            "The LLM-as-Judge tests every combination of chunking x retrieval",
            "strategies to find the champion. Below are all tunable parameters",
            "and their effects.",
            "",
        ]
        for param, desc in ExperimentRunner._PARAM_GUIDE.items():
            if isinstance(desc, dict):
                lines.append(f"{param}:")
                for k, v in desc.items():
                    lines.append(f"  {k}: {v}")
            else:
                lines.append(f"{param}: {desc}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def adjustment_rules() -> str:
        return """
=== Auto-Adjustment Rules (chunk x retrieval) ===

1. LOW SCORE (< 2.0) across ALL chunkings:
   - The document may not be well-parsed. Check bronze output.
   - Try different document loader strategy.

2. SMALL CHUNKS (256-500) perform WORSE:
   - The video confirmed: "token 256 and recursive 256 had worst performance"
   - Fix: increase chunk_size (1000-1500) for better context.

3. LARGE CHUNKS (> 1500) perform WORSE:
   - Context too noisy, dilutes relevant info.
   - Fix: decrease chunk_size or increase overlap.

4. STANDARD fails but MMR/HyDE succeed:
   - Retrieval needs diversity or hypothetical generation.
   - The video: "Hybrid search + HyDE went from 2/5 to 5/5"

5. MMR fails but STANDARD succeeds:
   - Queries are direct matches; diversity not needed.
   - Fix: use standard for speed.

6. HYDE is best but SLOW:
   - Accept the latency tradeoff for accuracy.
   - Try standard with a simpler dataset first.

7. PASS RATE < 50% on champion:
   - Review test_dataset.csv — questions may not match the corpus.
   - Ensure bronze + silver ran successfully.
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG Experiment Runner — test chunk x retrieval matrix"
    )
    parser.add_argument(
        "--mode",
        choices=["matrix", "auto-tune", "ingest-only"],
        default="matrix",
        help="What to run: matrix, auto-tune, or ingest-only",
    )
    parser.add_argument(
        "--config",
        default="rag_config.yaml",
        help="Path to RAG config YAML",
    )
    parser.add_argument(
        "--input",
        default="test_dataset.csv",
        help="Path to test dataset CSV",
    )
    parser.add_argument(
        "--bronze-json",
        default=None,
        help="Path to bronze JSON (auto-detected if not specified)",
    )
    parser.add_argument(
        "--guide",
        action="store_true",
        help="Print parameter guide and exit",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    if args.guide:
        print(ExperimentRunner.param_guide())
        print(ExperimentRunner.adjustment_rules())
        return 0

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set in .env")
        return 1

    runner = ExperimentRunner(
        config_path=args.config,
        test_csv=args.input,
        bronze_json=args.bronze_json,
    )

    if args.mode == "ingest-only":
        runner.load_experiment_config()
        runner.ensure_collections()
        logger.info("Ingestion complete. Collections: %s", runner.list_collections())
    elif args.mode == "auto-tune":
        runner.auto_tune()
    else:
        runner.run_matrix()
        champion, ranked = runner.find_champion()
        runner._print_report(champion, ranked)

    return 0


if __name__ == "__main__":
    sys.exit(main())
