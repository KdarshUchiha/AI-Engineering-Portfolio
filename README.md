# AI Engineering Portfolio

A growing collection of hands-on AI engineering projects, each one teaching and
demonstrating a core pattern used to build real, production-grade AI features.
Built while training toward a **Forward Deployed / Solutions Engineer** role.

Every project is:
- **Runnable** — clone, set one API key, run.
- **Readable** — the AI calls use the Anthropic SDK; the surrounding logic
  (retrieval, aggregation, tools) is plain Python so you can see exactly how it
  works, not hidden behind a framework.
- **Real** — solves an actual problem, not a toy demo.

## Projects

| # | Project | Core skill it demonstrates | Status |
|---|---------|----------------------------|--------|
| 1 | [Smart Document Q&A (RAG)](./01-rag-document-qa) | Retrieval-Augmented Generation: chunking, embeddings, vector similarity, grounded + cited answers | ✅ |
| 2 | [Real-Time Sentiment Dashboard](./02-sentiment-dashboard) | Structured outputs: forcing strict JSON from an LLM and computing on it; streaming aggregation | ✅ |
| 3 | [AI Customer-Support Agent](./03-support-agent) | Tool use / agents: Claude calls functions to look up real data, in a manual agent loop | ✅ |
| 4 | Meeting Notes & Action-Item Bot | Speech-to-text + summarization | 🔜 |
| 5 | SQL Copilot | Natural language → SQL → live results | 🔜 |
| 6 | Multimodal Product Inspector | Vision models | 🔜 |
| 7 | Personalized Learning Tutor | Agentic loops + memory/state | 🔜 |
| 8 | Code Review Assistant | Code understanding + GitHub API | 🔜 |
| 9 | Live Translation & Caption Tool | Low-latency streaming | 🔜 |
| 10 | AI Ops Copilot | RAG over logs/metrics | 🔜 |

## Tech

- **Python 3.12**
- **[Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python)** for Claude
- Standard library for everything else (no heavy ML deps to install)

## Quick start

```bash
git clone <this-repo-url>
cd ai-engineering-portfolio
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-your-key"   # get one at console.anthropic.com

# run any project
cd 01-rag-document-qa
python rag.py
```

Each project folder has its own `README.md` explaining what it does, how it
works, and what to learn from it.

## About

I'm building these to go deep on the patterns behind production AI systems —
RAG, structured outputs, tool-using agents, multimodal, and evaluation — and to
ship in public as I learn. Feedback and questions welcome.
