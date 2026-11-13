# Agentic RAG: Training Guide

A hands-on course that takes you from an empty folder to a working,
secure, document question-answering assistant built with **LangGraph**,
**FastAPI**, **Chainlit**, **ChromaDB / Qdrant**, and **OpenAI**.

Each chapter ends with a checkpoint: something you can run and see.

| # | Chapter | You will |
|---|---------|----------|
| 1 | [Overview and architecture](01-overview-and-architecture.md) | understand what the system does and how a question flows through it |
| 2 | [Setup](02-setup.md) | install everything and configure `.env` on Windows, macOS or Linux |
| 3 | [First run](03-first-run.md) | start the API and UI, upload a PDF, get a cited answer |
| 4 | [Feature guide](04-features.md) | tour every feature: roles, tabs, OCR engines, vision, diagnostics, admin, FinOps, security |
| 5 | [Code walkthrough](05-code-walkthrough.md) | read every module, from ingestion to UI |
| 6 | [Concepts](06-concepts.md) | learn the ideas behind RAG, hybrid search, reranking, LangGraph, RAGAS |
| 7 | [Evaluating answer quality](07-evaluation.md) | score answers in chat with RAGAS, and run golden-dataset evaluations |
| 8 | [Deployment and CI/CD](08-deployment-cicd.md) | run with Docker Compose, deploy to Cloud Run, automate with GitHub Actions |
| 9 | [Troubleshooting](09-troubleshooting.md) | fix the problems people actually hit |

**Who this is for:** you know basic Python and have used a terminal. You
don't need prior experience with LLMs, vector databases or LangGraph.

**Time:** about 2 hours for chapters 1 to 3 (most of it is installing).
