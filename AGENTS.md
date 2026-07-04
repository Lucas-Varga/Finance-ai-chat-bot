# FinanceChatBot

## Arquitetura (Medallion)

```
bronze/parse_text.py  ─→  silver_chunk_embed.py  ─→  gold_rag_agent.py
PDF → JSON paginado       JSON → chunks → ChromaDB    ChromaDB → retrieval → LLM
```

- **Bronze** (`parse_doc/parse_text.py`): extrai texto de PDFs via `UnstructuredLoader`, agrupa por página, salva em `data/bronze/parsed_pdfs/*.json`
- **Prata** (`silver_chunk_embed.py`): lê JSON do bronze, chunking (recursive/token), embeddings OpenAI, ingestão em ChromaDB (`chroma_db/`)
- **Ouro** (`gold_rag_agent.py`): carrega ChromaDB, retrieval (standard/MMR/HyDE), QA chain com LLM, CLI interativo
- **Avaliador** (`evaluator_llm_judge.py`): LLM-as-Judge, lê `rag_config.yaml` + `test_dataset.csv`

## Setup

```bash
source venv/bin/activate     # Python 3.14
```
`.env` precisa de `OPENAI_API_KEY=<chave>` (atualmente vazio).

`requirements.txt` gerado com `pip freeze`.

## Comandos

```bash
# Bronze — parse todos os PDFs em base/
make bronze

# Prata — chunk, embed, ingestar
make silver FILE=data/bronze/parsed_pdfs/doc1.json

# Ouro — CLI interativo
make gold

# Avaliar — executa evaluator LLM-as-Judge
make evaluate

# Ingerir todas as estratégias de chunking definidas no YAML
make ingest-all

# Testar matriz completa: chunking × retrieval
make matrix

# Auto-tune: encontra a melhor combinação (chunk+retrieval) e atualiza config
make auto-tune

# Ver configuração atual
make info

# Listar estratégias disponíveis
make list-strategies

# Ver último resultado da avaliação
make results

# Guia de parâmetros para LLM-as-Judge
python run_experiments.py --guide
python evaluator_llm_judge.py --guide
```

## Atenção (gotchas)

- **`OPENAI_API_KEY` vazia** no `.env` — sistema inteiro falha
- **`gold_rag_agent.py`** e **`silver_chunk_embed.py`** agora leem `rag_config.yaml` como default (CLI/env vars sobrescrevem)
- **Collection name** deriva de `{base_collection_name}_{chunk_strategy}` (lido do YAML). Default: `finance_docs_recursive`
- **ChromaDB** em `chroma_db/` está no `.gitignore`
- **Só 1 PDF** na `base/` (`doc1.pdf` — JDBC Java, não finanças)
- **Sem testes automatizados** — sem pytest, sem CI

## Convenções de código

- **Factory Pattern** para chunking (`SplitterFactory`) e retrieval (`RetrieverFactory`)
- **Dataclasses frozen** para config (`AgentConfig`, `ETLConfig`)
- **Structured logging** em JSON (ouro) e texto plano (bronze/prata)
- **`from __future__ import annotations`** no topo de todos os módulos
- **CLI** do ouro tem comandos `/strategy`, `/k`, `/info`, `/help`, `/quit`

## Estrutura de dados

| Arquivo | Função |
|---|---|
| `base/` | PDFs fonte (input do bronze) |
| `data/bronze/parsed_pdfs/` | JSON paginado (output do bronze, input da prata) |
| `chroma_db/` | Vector store ChromaDB persistido |
| `test_dataset.csv` | 7 pares pergunta/resposta esperada |
| `evaluation_results.csv` | Resultados da última avaliação |
| `logs/` | Logs do bronze e ouro |
| `run_experiments.py` | Runner de experimentos batch (testa estratégias, auto-tune) |
| `Makefile` | Interface de comandos do pipeline |
| `rag_config.yaml` | Configuração central do RAG |
| `.env` | OPENAI_API_KEY (obrigatório) |

## Time de desenvolvimento

```
Tech Lead (coordenador)
├── Backend Python Dev  — pipeline RAG (langchain-rag skill)
├── Frontend Dev        — interfaces web (frontend-design skill)
└── QA Engineer         — testes e avaliação
```

Configurações dos subagentes em `.opencode/agents/`. O `opencode.json` mapeia os atalhos `backend`, `frontend`, `qa` para cada agente.

## Skills instaladas

### YouTube Transcript (`.agents/skills/youtube-transcript/`)
Extrair transcrições de vídeos via `youtube-transcript-api`. Script esperado: `scripts/get_transcript.py` (requer `uv`).

### LangChain RAG (`.agents/skills/langchain-rag/`)
Cobertura completa de RAG: document loaders, text splitters, embeddings, vector stores, retrieval, RAG Agent. Ativado automaticamente ao construir sistemas RAG.
