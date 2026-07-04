# Team — FinanceChatBot

## Estrutura

```
Tech Lead (Coordenador)
├── Backend Python Dev (langchain-rag skill)
├── Frontend Dev (frontend-design skill)
└── QA Engineer
```

## Responsabilidades

### Tech Lead (Quem está lendo este documento)
- Coordena o time e define prioridades
- Revisa e aprova implementações
- Garante que a arquitetura Medallion seja seguida
- Traduz requisitos de negócio em tarefas técnicas

### Backend Python Dev (`.opencode/agents/backend-python.json`)
- Camadas Bronze, Prata, Ouro
- Chunking, embeddings, retrieval
- ChromaDB, OpenAI, LangChain
- **Skill**: langchain-rag

### Frontend Dev (`.opencode/agents/frontend-dev.json`)
- Interface web para o RAG Agent
- Dashboard de métricas e avaliação
- **Skill**: frontend-design

### QA Engineer (`.opencode/agents/qa-engineer.json`)
- Testes automatizados e dataset de teste
- LLM-as-Judge evaluation
- Relatórios de qualidade

## Workflow

1. **Tech Lead** abre uma task descrevendo o que precisa ser feito
2. O subagente responsável implementa
3. **Tech Lead** revisa o resultado
4. Se aprovado, marca como concluído

## Skills do repositório

| Skill | Localização | Para quem |
|---|---|---|
| LangChain RAG | `.agents/skills/langchain-rag/` | Backend Dev |
| Frontend Design | `.agents/skills/frontend-design/` | Frontend Dev |
| YouTube Transcript | `.agents/skills/youtube-transcript/` | Qualquer membro |
