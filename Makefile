.PHONY: bronze silver gold evaluate test-retrieval test-agent auto-tune info list-strategies clean help ingest-all matrix

VENV    = venv
PYTHON  = $(VENV)/bin/python3
CONFIG  = rag_config.yaml
TEST_CSV = test_dataset.csv
RESULTS  = evaluation_results.csv

help:
	@echo "=== FinanceChatBot - Makefile ==="
	@echo ""
	@echo "Pipeline:"
	@echo "  make bronze        Parse todos os PDFs em base/ -> JSON paginado"
	@echo "  make silver FILE=  Chunk, embed e ingesta no ChromaDB (ex: FILE=data/bronze/parsed_pdfs/doc1.json)"
	@echo "  make gold          CLI interativo do RAG Agent"
	@echo ""
	@echo "Testes e Avaliacao:"
	@echo "  make evaluate      Executa LLM-as-Judge com config atual"
	@echo "  make test-retrieval Testa TODAS as estrategias de retrieval em batch"
	@echo "  make test-agent     Testa o agente completo (retrieval + geracao)"
	@echo ""
	@echo "Auto-Ajuste:"
	@echo "  make auto-tune     Testa todas estrategias, encontra a campea, atualiza config"
	@echo ""
	@echo "Informacao:"
	@echo "  make info          Mostra configuracao atual"
	@echo "  make list-strategies  Lista estrategias disponiveis"
	@echo "  make list-collections Lista colecoes no ChromaDB"
	@echo "  make results       Mostra ultimos resultados da avaliacao"
	@echo ""
	@echo "Utilitarios:"
	@echo "  make clean         Limpa cache e temporarios"
	@echo "  make reqs          Gera requirements.txt atualizado"
	@echo ""

# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------

bronze:
	$(PYTHON) parse_doc/parse_text.py

silver:
	$(PYTHON) silver_chunk_embed.py $(FILE)

gold:
	$(PYTHON) gold_rag_agent.py

# -------------------------------------------------------------------
# Testes e Avaliacao
# -------------------------------------------------------------------

evaluate:
	$(PYTHON) evaluator_llm_judge.py --config $(CONFIG) --input $(TEST_CSV) --output $(RESULTS)

# -------------------------------------------------------------------
# Experimentos — Matriz chunking × retrieval
# -------------------------------------------------------------------

# Cria todas as collections definidas em rag_config.yaml (experiment.chunking_strategies)
ingest-all:
	$(PYTHON) run_experiments.py --mode ingest-only --config $(CONFIG)

# Testa matriz completa: chunk × retrieval
matrix:
	$(PYTHON) run_experiments.py --mode matrix --config $(CONFIG) --input $(TEST_CSV)

# Auto-ajuste: testa matriz, encontra campeã, atualiza config
auto-tune:
	$(PYTHON) run_experiments.py --mode auto-tune --config $(CONFIG) --input $(TEST_CSV)

# Mantido por compatibilidade — executa matrix
test-retrieval: matrix

test-agent:
	@echo "Teste de agente completo sera implementado na Fase 3"
	@echo "Por enquanto, use: make auto-tune"

# -------------------------------------------------------------------
# Informacao
# -------------------------------------------------------------------

info:
	@echo "=== Current RAG Configuration ==="
	@cat $(CONFIG)

list-strategies:
	@echo "=== Estrategias de Chunking ==="
	@echo "  recursive   - RecursiveCharacterTextSplitter (padrao)"
	@echo "  token       - TokenTextSplitter (baseado em tokens)"
	@echo ""
	@echo "=== Estrategias de Retrieval ==="
	@echo "  standard    - Similaridade por cosseno (dense)"
	@echo "  mmr         - Maximal Marginal Relevance (diversidade)"
	@echo "  hyde        - Hypothetical Document Embeddings"
	@echo ""
	@echo "=== Parametros Ajustaveis ==="
	@echo "  chunk_size     - Tamanho do chunk (default: 1000)"
	@echo "  chunk_overlap  - Sobreposicao entre chunks (default: 200)"
	@echo "  top_k          - Numero de documentos recuperados (default: 4)"
	@echo "  temperature    - Temperatura do LLM (default: 0.1)"
	@echo "  strategy       - Estrategia de retrieval (standard|mmr|hyde)"

list-collections:
	$(PYTHON) -c "import chromadb; c=chromadb.PersistentClient(path='./chroma_db'); [print(f'  {x.name}: {x.count()} docs') for x in c.list_collections()]"

results:
	@if [ -f $(RESULTS) ]; then echo "=== Last Evaluation Results ===" && python3 -c "import pandas; df=pandas.read_csv('$(RESULTS)'); print(f'Avg Context Relevance: {df.context_relevance_score.mean():.2f}/5'); print(f'Avg Answer Correctness: {df.answer_correctness_score.mean():.2f}/5'); print(f'Pass Rate (score>=3): {(df.answer_correctness_score>=3).mean()*100:.0f}%')"; else echo "No evaluation results found. Run 'make evaluate' first."; fi

# -------------------------------------------------------------------
# Utilitarios
# -------------------------------------------------------------------

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

reqs:
	$(PYTHON) -m pip freeze > requirements.txt
	@echo "requirements.txt atualizado"
