#!/usr/bin/env python3
"""
LLM-as-Judge Evaluator for RAG Systems.

Evaluates the RAG pipeline's retrieval and generation quality using an LLM
as an impartial judge. Supports CLI arguments, pass/fail metrics, and
structured output.

Usage:
    python evaluator_llm_judge.py
    python evaluator_llm_judge.py --config rag_config.yaml --input test_dataset.csv
    python evaluator_llm_judge.py --guide

This evaluator IS the LLM-as-Judge for the RAG system. It produces scores
(0-5) for context relevance and answer correctness, plus a pass/fail
classification used by run_experiments.py for auto-tuning.
"""

import argparse
import os
import sys
from typing import List, Optional

import yaml
import pandas as pd
import logging
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from gold_rag_agent import GoldRAGAgent, AgentConfig, RetrieverStrategy

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LLMJudgeEvaluator")


# ---------------------------------------------------------------------------
# Evaluation Schema
# ---------------------------------------------------------------------------

SCORE_THRESHOLD_PASS = 3


class EvaluationResult(BaseModel):
    """Schema returned by the LLM Judge for each Q&A pair."""
    context_relevance_score: int = Field(
        ..., description="Score from 0 to 5 on how relevant the retrieved context is to the question."
    )
    context_relevance_reasoning: str = Field(
        ..., description="Detailed explanation for the context relevance score."
    )
    answer_correctness_score: int = Field(
        ..., description="Score from 0 to 5 on how accurate the answer is compared to the expected answer."
    )
    answer_correctness_reasoning: str = Field(
        ..., description="Detailed explanation for the answer correctness score."
    )

    def is_pass(self) -> bool:
        """A test passes if both scores are >= SCORE_THRESHOLD_PASS."""
        return (
            self.context_relevance_score >= SCORE_THRESHOLD_PASS
            and self.answer_correctness_score >= SCORE_THRESHOLD_PASS
        )

    def is_context_pass(self) -> bool:
        return self.context_relevance_score >= SCORE_THRESHOLD_PASS

    def is_answer_pass(self) -> bool:
        return self.answer_correctness_score >= SCORE_THRESHOLD_PASS


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    LLM-as-Judge evaluator for the RAG system.

    Evaluates each question in the test dataset by:
    1. Getting the RAG agent's response (context + answer)
    2. Sending (question, context, answer, expected_answer) to the Judge LLM
    3. Receiving structured scores (0-5) with reasoning
    4. Computing pass/fail and aggregate metrics
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        raw_config = self._load_config(config_path)

        silver_cfg = raw_config.get("silver_layer", {})
        gold_cfg = raw_config.get("gold_layer", {})

        base_collection = silver_cfg.get("vector_db", {}).get("base_collection_name", "finance_docs")
        chunk_strategy = silver_cfg.get("chunking", {}).get("strategy", "recursive")
        collection_name = f"{base_collection}_{chunk_strategy}"

        agent_config = AgentConfig(
            chroma_path=silver_cfg.get("vector_db", {}).get("path", "./chroma_db"),
            collection_name=collection_name,
            embedding_model=silver_cfg.get("embedding", {}).get("model", "text-embedding-3-small"),
            llm_model=gold_cfg.get("llm", {}).get("model", "gpt-4o-mini"),
            temperature=gold_cfg.get("llm", {}).get("temperature", 0.1),
            strategy=RetrieverStrategy(gold_cfg.get("retrieval", {}).get("strategy", "standard")),
            k=gold_cfg.get("retrieval", {}).get("top_k", 4),
        )

        self.agent = GoldRAGAgent(config=agent_config)
        self.judge_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        ).with_structured_output(EvaluationResult)

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.\n"
                "Your task is to assess the quality of the retrieved context and the generated answer.\n\n"
                "Evaluation Criteria:\n"
                "1. Context Relevance (0-5): Does the retrieved context contain the information needed?\n"
                "2. Answer Correctness (0-5): Does the answer match the expected answer? No hallucinations?\n\n"
                "A score >= 3 means PASS. Provide both scores with detailed reasoning."
            ),
            (
                "human",
                "Question: {question}\n"
                "Retrieved Context: {context}\n"
                "Generated Answer: {generated_answer}\n"
                "Expected Answer: {expected_answer}"
            ),
        ])

    def _load_config(self, path: str) -> dict:
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error("Failed to load config from %s: %s", path, e)
            raise

    def evaluate_row(self, question: str, expected_answer: str) -> dict:
        """Evaluate a single Q&A pair and return scores + pass/fail."""
        logger.info("Evaluating: %s...", question[:60])

        agent_response = self.agent.ask(question)
        generated_answer = agent_response.get("answer", "")
        context_docs = agent_response.get("sources", [])
        context_text = "\n".join(
            [doc.get("content", "") for doc in context_docs]
        ) if context_docs else "No context retrieved."

        chain = self.prompt | self.judge_llm
        evaluation: EvaluationResult = chain.invoke({
            "question": question,
            "context": context_text,
            "generated_answer": generated_answer,
            "expected_answer": expected_answer,
        })

        return {
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "context_relevance_score": evaluation.context_relevance_score,
            "context_relevance_reasoning": evaluation.context_relevance_reasoning,
            "context_pass": evaluation.is_context_pass(),
            "answer_correctness_score": evaluation.answer_correctness_score,
            "answer_correctness_reasoning": evaluation.answer_correctness_reasoning,
            "answer_pass": evaluation.is_answer_pass(),
            "overall_pass": evaluation.is_pass(),
        }

    def run_evaluation(
        self,
        csv_path: str,
        output_path: str = "evaluation_results.csv",
    ) -> pd.DataFrame:
        """Run evaluation on all questions in the CSV and save results."""
        df = pd.read_csv(csv_path)
        required_cols = {'question', 'expected_answer'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        results = []
        for _, row in df.iterrows():
            try:
                res = self.evaluate_row(row['question'], row['expected_answer'])
                results.append(res)
            except Exception as e:
                logger.error("Error evaluating row: %s", e)

        results_df = pd.DataFrame(results)

        # Aggregate metrics
        avg_relevance = results_df['context_relevance_score'].mean()
        avg_correctness = results_df['answer_correctness_score'].mean()
        pass_rate = results_df['overall_pass'].mean()

        logger.info("")
        logger.info("=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)
        logger.info("Average Context Relevance: %.2f/5", avg_relevance)
        logger.info("Average Answer Correctness: %.2f/5", avg_correctness)
        logger.info("Overall Pass Rate: %.0f%%", pass_rate * 100)
        logger.info("Total Questions: %d", len(results_df))
        logger.info("=" * 50)

        results_df.to_csv(output_path, index=False)
        logger.info("Detailed results saved to %s", output_path)

        return results_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge Evaluator for RAG Systems"
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
        "--output",
        default="evaluation_results.csv",
        help="Path to output CSV",
    )
    parser.add_argument(
        "--guide",
        action="store_true",
        help="Print evaluation guide and exit",
    )
    return parser.parse_args()


def print_guide() -> None:
    """Print documentation for the evaluator / LLM-as-Judge system."""
    print("""
=== LLM-as-Judge Evaluation Guide ===

The LLM-as-Judge evaluates RAG responses on two axes:

1. CONTEXT RELEVANCE (0-5):
   5 = Perfect context, directly answers the question
   4 = Good context, minor irrelevant details
   3 = Passable context, some key info present
   2 = Poor context, missing important information
   1 = Mostly irrelevant context
   0 = No relevant context retrieved

2. ANSWER CORRECTNESS (0-5):
   5 = Perfect answer, matches expected, no hallucination
   4 = Good answer, minor omissions
   3 = Passable answer, core information present
   2 = Poor answer, missing key facts
   1 = Mostly incorrect
   0 = Completely wrong or "I don't know"

PASS / FAIL: Score >= 3 on BOTH axes = PASS

=== Auto-Adjust Trigger ===

Based on scores:
- avg_score < 2.0: Critical failure — change retrieval strategy
- avg_score 2.0-3.5: Suboptimal — try different strategy or parameters
- avg_score > 3.5: Good — minor refinements only
- pass_rate < 50%: Systematically failing — need major changes

Adjustable parameters (in rag_config.yaml):
  gold_layer.retrieval.strategy: standard | mmr | hyde
  gold_layer.retrieval.top_k: 1-10
  gold_layer.llm.temperature: 0.0-1.0
  silver_layer.chunking.chunk_size: 500-2000
  silver_layer.chunking.chunk_overlap: 50-500

Run 'python run_experiments.py --mode auto-tune --update-config' to
automatically find and apply the best strategy.
""")


def main() -> int:
    args = parse_args()

    if args.guide:
        print_guide()
        return 0

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set in .env")
        return 1

    config_path = args.config
    input_csv = args.input

    if not os.path.exists(config_path):
        logger.error("Config file %s not found.", config_path)
        return 1
    if not os.path.exists(input_csv):
        logger.error("Input CSV %s not found.", input_csv)
        return 1

    evaluator = RAGEvaluator(config_path)
    evaluator.run_evaluation(input_csv, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
