#!/usr/bin/env python3
"""
Gold Layer - Retrieval and Generation (RAG Agent)

Production-grade RAG agent that loads the ChromaDB vector store created in the
Silver layer, exposes multiple retrieval strategies (standard, MMR, HyDE), and
builds a LangChain Expression Language (LCEL) QA chain with an interactive CLI.
"""

from __future__ import annotations

import argparse
import logging
import logging.config
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "structured": {
            "format": (
                "{\"time\": \"%(asctime)s\", \"level\": \"%(levelname)s\", "
                "\"logger\": \"%(name)s\", \"message\": \"%(message)s\", "
                "\"module\": \"%(module)s\", \"func\": \"%(funcName)s\", "
                "\"line\": %(lineno)d}"
            ),
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "structured",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "structured",
            "filename": "logs/gold_rag_agent.log",
            "maxBytes": 10485760,
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "gold_rag": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
    "root": {
        "level": "WARNING",
        "handlers": ["console"],
    },
}


def configure_logging() -> logging.Logger:
    """Configure structured logging and return the application logger."""
    os.makedirs("logs", exist_ok=True)
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger("gold_rag")


logger = configure_logging()


# ---------------------------------------------------------------------------
# Enums & Dataclasses
# ---------------------------------------------------------------------------


class RetrieverStrategy(str, Enum):
    STANDARD = "standard"
    MMR = "mmr"
    HYDE = "hyde"


def _load_gold_config_from_yaml(config_path: str = "rag_config.yaml") -> Dict[str, Any]:
    """Load default config from rag_config.yaml, falling back to built-in defaults."""
    defaults: Dict[str, Any] = {
        "chroma_path": "./chroma_db",
        "embedding_model": "text-embedding-3-small",
        "llm_model": "gpt-4o-mini",
        "temperature": 0.1,
        "strategy": "standard",
        "k": 4,
    }
    if not os.path.exists(config_path):
        return defaults
    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        silver = cfg.get("silver_layer", {})
        gold = cfg.get("gold_layer", {})
        retrieval = gold.get("retrieval", {})
        llm = gold.get("llm", {})
        vector_db = silver.get("vector_db", {})
        chunking = silver.get("chunking", {})

        defaults["chroma_path"] = vector_db.get("path", defaults["chroma_path"])
        defaults["embedding_model"] = silver.get("embedding", {}).get("model", defaults["embedding_model"])
        defaults["llm_model"] = llm.get("model", defaults["llm_model"])
        defaults["temperature"] = llm.get("temperature", defaults["temperature"])
        defaults["strategy"] = retrieval.get("strategy", defaults["strategy"])
        defaults["k"] = retrieval.get("top_k", defaults["k"])

        # Derive collection name like evaluator does
        base_collection = vector_db.get("base_collection_name", "finance_docs")
        chunk_strategy = chunking.get("strategy", "recursive")
        defaults["collection_name"] = f"{base_collection}_{chunk_strategy}"
    except Exception:
        logger.warning("Failed to load %s, using built-in defaults", config_path)
        defaults["collection_name"] = defaults.get("collection_name", "finance_docs_recursive")
    return defaults


@dataclass(frozen=True)
class AgentConfig:
    """Runtime configuration for the Gold RAG agent."""

    chroma_path: str = "data/chroma"
    collection_name: str = "silver_collection"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.1
    strategy: RetrieverStrategy = RetrieverStrategy.STANDARD
    k: int = 4
    fetch_k: int = 20
    lambda_mult: float = 0.5
    chunk_prefix: str = "Answer the following question based on the context:\n\n"


# ---------------------------------------------------------------------------
# HyDE Helper
# ---------------------------------------------------------------------------

HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert technical writer. Generate a concise, plausible "
            "hypothetical document that would contain the answer to the user's "
            "question. Do not mention that it is hypothetical. Output only the "
            "document text.",
        ),
        ("user", "{question}"),
    ]
)


class HyDERetriever(BaseRetriever):
    """
    Hypothetical Document Embeddings (HyDE) retriever.

    Generates a hypothetical answer document for the user question, embeds it,
    and uses the resulting embedding to retrieve semantically similar chunks.
    """

    vectorstore: Chroma
    llm: BaseChatModel
    k: int = 4
    fetch_k: int = 20
    lambda_mult: float = 0.5

    class Config:
        arbitrary_types_allowed = True

    def _generate_hypothetical_document(self, question: str) -> str:
        """Generate a hypothetical document for the given question."""
        chain = HYDE_PROMPT | self.llm | StrOutputParser()
        try:
            hypothetical = chain.invoke({"question": question})
            logger.debug(
                "HyDE hypothetical document generated",
                extra={"question": question, "hypothetical_len": len(hypothetical)},
            )
            return hypothetical
        except Exception:
            logger.exception("HyDE generation failed; falling back to raw question")
            return question

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> List[Document]:
        hypothetical_doc = self._generate_hypothetical_document(query)
        docs = self.vectorstore.similarity_search_with_relevance_scores(
            hypothetical_doc, k=self.k
        )
        results: List[Document] = []
        for doc, score in docs:
            doc.metadata["hyde_score"] = float(score)
            doc.metadata["hyde_generated"] = True
            results.append(doc)
        logger.info(
            "HyDE retrieval completed",
            extra={"query": query, "num_docs": len(results)},
        )
        return results

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


# ---------------------------------------------------------------------------
# Retriever Factory
# ---------------------------------------------------------------------------


class RetrieverFactory:
    """
    Factory that builds retrievers for the Gold layer based on a strategy.

    Supported strategies:
      - standard: dense similarity search
      - mmr: Maximal Marginal Relevance for diversity
      - hyde: Hypothetical Document Embeddings
    """

    def __init__(
        self,
        vectorstore: Chroma,
        llm: Optional[BaseChatModel] = None,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> None:
        self.vectorstore = vectorstore
        self.llm = llm
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult

    @classmethod
    def from_config(
        cls,
        config: AgentConfig,
        vectorstore: Chroma,
        llm: Optional[BaseChatModel] = None,
    ) -> "RetrieverFactory":
        return cls(
            vectorstore=vectorstore,
            llm=llm,
            k=config.k,
            fetch_k=config.fetch_k,
            lambda_mult=config.lambda_mult,
        )

    def build(self, strategy: RetrieverStrategy) -> BaseRetriever:
        """Build and return a retriever for the requested strategy."""
        logger.info(
            "Building retriever",
            extra={"strategy": strategy.value, "k": self.k, "fetch_k": self.fetch_k},
        )

        if strategy == RetrieverStrategy.STANDARD:
            return self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.k},
            )

        if strategy == RetrieverStrategy.MMR:
            return self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self.k,
                    "fetch_k": self.fetch_k,
                    "lambda_mult": self.lambda_mult,
                },
            )

        if strategy == RetrieverStrategy.HYDE:
            if self.llm is None:
                raise ValueError("HyDE strategy requires an LLM instance.")
            return HyDERetriever(
                vectorstore=self.vectorstore,
                llm=self.llm,
                k=self.k,
                fetch_k=self.fetch_k,
                lambda_mult=self.lambda_mult,
            )

        raise ValueError(f"Unsupported retriever strategy: {strategy}")


# ---------------------------------------------------------------------------
# QA Chain Builder (LCEL)
# ---------------------------------------------------------------------------

QA_SYSTEM_PROMPT = (
    "You are a knowledgeable, precise assistant for a retrieval-augmented "
    "generation system. Use ONLY the provided context to answer the question. "
    "If the context does not contain enough information, say you do not know. "
    "Cite source document identifiers when available. Be concise and factual."
)

QA_USER_PROMPT = """
Context:
{context}

Question: {question}

Answer:"""


def format_documents(docs: List[Document]) -> str:
    """Format retrieved documents into a single context string."""
    if not docs:
        return "(No relevant context found.)"
    formatted: List[str] = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        loc = f" (source: {source}" + (f", page: {page}" if page else "") + ")"
        formatted.append(f"[{idx}]{loc}\n{doc.page_content}")
    return "\n\n".join(formatted)


class RAGChainBuilder:
    """Builds the LCEL QA chain connecting retriever and LLM."""

    def __init__(self, retriever: BaseRetriever, llm: BaseChatModel) -> None:
        self.retriever = retriever
        self.llm = llm

    def build(self) -> Runnable:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QA_SYSTEM_PROMPT),
                ("user", QA_USER_PROMPT),
            ]
        )

        chain = (
            {
                "context": self.retriever | format_documents,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        logger.info("LCEL QA chain constructed")
        return chain


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class GoldRAGAgent:
    """High-level orchestrator for the Gold RAG layer."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._validate_environment()

        logger.info(
            "Initializing GoldRAGAgent",
            extra={
                "chroma_path": config.chroma_path,
                "collection": config.collection_name,
                "strategy": config.strategy.value,
            },
        )

        self.embeddings = OpenAIEmbeddings(model=config.embedding_model)
        self.vectorstore = self._load_vectorstore()
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature,
        )

        self.factory = RetrieverFactory.from_config(
            config=config,
            vectorstore=self.vectorstore,
            llm=self.llm,
        )
        self.retriever = self.factory.build(config.strategy)
        self.chain = RAGChainBuilder(self.retriever, self.llm).build()

        logger.info("GoldRAGAgent ready", extra={"strategy": config.strategy.value})

    def _validate_environment(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY not found in environment")
            raise EnvironmentError("OPENAI_API_KEY is required but not set.")

    def _load_vectorstore(self) -> Chroma:
        if not os.path.isdir(self.config.chroma_path):
            logger.error(
                "ChromaDB path does not exist",
                extra={"path": self.config.chroma_path},
            )
            raise FileNotFoundError(
                f"ChromaDB not found at {self.config.chroma_path}. "
                "Run the Silver layer ingestion first."
            )
        logger.info(
            "Loading ChromaDB vector store",
            extra={"path": self.config.chroma_path, "collection": self.config.collection_name},
        )
        return Chroma(
            persist_directory=self.config.chroma_path,
            embedding_function=self.embeddings,
            collection_name=self.config.collection_name,
        )

    def ask(self, question: str) -> Dict[str, Any]:
        """Answer a question and return the answer plus retrieved sources."""
        if not question.strip():
            return {"answer": "", "sources": []}

        logger.info("Processing question", extra={"question": question})
        try:
            retrieved_docs = self.retriever.invoke(question)
            answer = self.chain.invoke(question)
            sources = [
                {
                    "content": doc.page_content[:300],
                    "metadata": doc.metadata,
                }
                for doc in retrieved_docs
            ]
            logger.info(
                "Answer generated",
                extra={"num_sources": len(sources), "answer_len": len(answer)},
            )
            return {"answer": answer, "sources": sources}
        except Exception as exc:
            logger.exception("Failed to answer question", extra={"question": question})
            return {"answer": f"[ERROR] {exc}", "sources": []}

    def switch_strategy(self, strategy: RetrieverStrategy) -> None:
        """Switch retrieval strategy at runtime and rebuild the chain."""
        logger.info("Switching retriever strategy", extra={"new_strategy": strategy.value})
        self.config = AgentConfig(
            chroma_path=self.config.chroma_path,
            collection_name=self.config.collection_name,
            embedding_model=self.config.embedding_model,
            llm_model=self.config.llm_model,
            temperature=self.config.temperature,
            strategy=strategy,
            k=self.config.k,
            fetch_k=self.config.fetch_k,
            lambda_mult=self.config.lambda_mult,
            chunk_prefix=self.config.chunk_prefix,
        )
        self.retriever = self.factory.build(strategy)
        self.chain = RAGChainBuilder(self.retriever, self.llm).build()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

HELP_TEXT = """
Gold RAG Agent - Interactive CLI
---------------------------------
Commands:
  /strategy <standard|mmr|hyde>  Switch retrieval strategy
  /k <int>                        Set top-k (rebuilds retriever)
  /info                           Show current configuration
  /help                           Show this help
  /quit | /exit                   Exit the agent

Otherwise, type your question and press Enter.
"""


def parse_args() -> AgentConfig:
    yaml_defaults = _load_gold_config_from_yaml()

    parser = argparse.ArgumentParser(
        description="Gold Layer RAG Agent (Retrieval & Generation)"
    )
    parser.add_argument(
        "--chroma-path",
        default=os.getenv("CHROMA_PATH", yaml_defaults["chroma_path"]),
        help="Path to the ChromaDB persisted in the Silver layer.",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("CHROMA_COLLECTION", yaml_defaults["collection_name"]),
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", yaml_defaults["embedding_model"]),
        help="OpenAI embedding model name.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("LLM_MODEL", yaml_defaults["llm_model"]),
        help="OpenAI chat model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("LLM_TEMPERATURE", str(yaml_defaults["temperature"]))),
        help="LLM sampling temperature.",
    )
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in RetrieverStrategy],
        default=os.getenv("RETRIEVER_STRATEGY", yaml_defaults["strategy"]),
        help="Retrieval strategy to use.",
    )
    parser.add_argument("--k", type=int, default=yaml_defaults["k"], help="Top-k documents to retrieve.")
    parser.add_argument(
        "--fetch-k", type=int, default=20, help="Candidate set size for MMR."
    )
    parser.add_argument(
        "--lambda-mult",
        type=float,
        default=0.5,
        help="Diversity vs relevance tradeoff for MMR (0-1).",
    )
    args = parser.parse_args()

    return AgentConfig(
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        temperature=args.temperature,
        strategy=RetrieverStrategy(args.strategy),
        k=args.k,
        fetch_k=args.fetch_k,
        lambda_mult=args.lambda_mult,
    )


def print_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        print("\n[No sources retrieved]")
        return
    print("\n--- Sources ---")
    for idx, src in enumerate(sources, start=1):
        meta = src.get("metadata", {})
        source = meta.get("source", "unknown")
        page = meta.get("page", "")
        loc = f"source={source}" + (f", page={page}" if page != "" else "")
        print(f"[{idx}] {loc}")
        print(f"    {src.get('content', '')[:160]}...")
    print("---------------\n")


def interactive_loop(agent: GoldRAGAgent) -> None:
    print(HELP_TEXT)
    print(f"Active strategy: {agent.config.strategy.value} | k={agent.config.k}\n")

    while True:
        try:
            user_input = input("ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit"):
                print("Goodbye!")
                break
            if cmd == "/help":
                print(HELP_TEXT)
                continue
            if cmd == "/info":
                print(
                    f"strategy={agent.config.strategy.value} "
                    f"k={agent.config.k} fetch_k={agent.config.fetch_k} "
                    f"lambda_mult={agent.config.lambda_mult} "
                    f"llm={agent.config.llm_model} "
                    f"embedding={agent.config.embedding_model}"
                )
                continue
            if cmd == "/strategy":
                try:
                    new_strategy = RetrieverStrategy(arg.lower())
                    agent.switch_strategy(new_strategy)
                    print(f"Strategy switched to: {new_strategy.value}")
                except ValueError:
                    print(
                        f"Invalid strategy '{arg}'. "
                        f"Choose: {', '.join(s.value for s in RetrieverStrategy)}"
                    )
                continue
            if cmd == "/k":
                try:
                    new_k = int(arg)
                    if new_k <= 0:
                        raise ValueError
                    agent.config = AgentConfig(
                        chroma_path=agent.config.chroma_path,
                        collection_name=agent.config.collection_name,
                        embedding_model=agent.config.embedding_model,
                        llm_model=agent.config.llm_model,
                        temperature=agent.config.temperature,
                        strategy=agent.config.strategy,
                        k=new_k,
                        fetch_k=agent.config.fetch_k,
                        lambda_mult=agent.config.lambda_mult,
                        chunk_prefix=agent.config.chunk_prefix,
                    )
                    agent.factory.k = new_k
                    agent.retriever = agent.factory.build(agent.config.strategy)
                    agent.chain = RAGChainBuilder(agent.retriever, agent.llm).build()
                    print(f"k updated to: {new_k}")
                except ValueError:
                    print("k must be a positive integer.")
                continue

            print(f"Unknown command: {cmd}. Type /help for options.")
            continue

        result = agent.ask(user_input)
        print("\nAnswer:")
        print(result["answer"])
        print_sources(result["sources"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    load_dotenv()
    config = parse_args()

    try:
        agent = GoldRAGAgent(config)
    except (EnvironmentError, FileNotFoundError) as exc:
        logger.error("Agent initialization failed", extra={"error": str(exc)})
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected initialization error")
        print(f"[FATAL] Unexpected error: {exc}", file=sys.stderr)
        return 1

    try:
        interactive_loop(agent)
    except Exception:
        logger.exception("Fatal error during interactive loop")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())