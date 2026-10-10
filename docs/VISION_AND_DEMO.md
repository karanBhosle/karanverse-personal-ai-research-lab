# Vision and demo data

This document explains **who karanVerse Research Lab is for** and **which real data** should appear in README screenshots and local demos.

## Who this app is for

**Primary user:** Karan Bhosle — Data Scientist building personal research infrastructure.

**Goals:**

- Ground answers in **your** notes, project docs, and papers—not generic web chat.
- Run **evidence-first** research: plan → literature + local retrieval → critique → cited report.
- Keep **personal knowledge** separate from public corpus unless you explicitly widen scope.
- Iterate on **hybrid retrieval**, **agentic RAG**, and **MCP-style tooling** in one local lab.

The product is a **single-researcher lab**, not multi-tenant SaaS. The UI and copy assume you are improving your own stack (this repo is both the product and a showcased project).

## Canonical demo seed

| Source | Path | What it contains |
|--------|------|------------------|
| Portfolio seed | [`data/personal/portfolio_seed.json`](../data/personal/portfolio_seed.json) | Profile, **karanVerse Research Lab** project, portfolio app, learning notes (Graph RAG, hybrid retrieval, agents/MCP) |
| Import API | `POST /knowledge/import/portfolio` | Ingests seed into `data/processed/` with scopes `PERSONAL_KNOWLEDGE` and `PROJECT` |
| Research history | `data/research_history/*.json` | Past workflow runs (shown on Dashboard, History, Graph) |
| Public sample | Ingested PDFs / notes under `PUBLIC_RESEARCH` | Hybrid search demos alongside personal scope |

### Example queries that match the vision

| Screen | Example query | Why |
|--------|----------------|-----|
| Knowledge | *What have I learned about Graph RAG?* | Triggers **personal + project** scope inference; hits learning notes in the seed |
| Knowledge gaps | *What don't I know about connecting Neo4j to hybrid retrieval?* | Mixes history + corpus gap analysis |
| Research | *How should my personal research lab combine hybrid retrieval with agentic workflows?* | Aligns with lab description and focus areas |
| Graph | Filter by research / concept nodes | Visualizes history + ingested knowledge |

Scope inference lives in [`backend/app/retrieval/knowledge_scope.py`](../backend/app/retrieval/knowledge_scope.py). Queries with *“I’ve learned”*, *“my projects”*, etc. search personal/project corpora; generic technical questions default to **public research only**.

## Refresh demo data before screenshots

From the repo root (API need not be running for the script’s Python import path):

```bash
./scripts/seed_demo_data.sh
```

Or manually:

```bash
curl -X POST http://localhost:8000/knowledge/import/portfolio
```

Then open the UI, run the example searches above, and capture pages.

## Editing screenshots

Yes—you can change them in two ways:

1. **Replace the composite** [`docs/screenshots/ui-overview.jpg`](screenshots/ui-overview.jpg) (or add per-page PNGs if you split the README tour again).
2. **Regenerate from the running app** (recommended so data matches seed + history):
   - Start API + frontend (see root README).
   - Seed portfolio (above).
   - Visit each route, use the example queries, then screenshot (browser devtools, Playwright, or Cursor browser tools).

When you update screenshots, adjust the **caption bullets** under “UI tour” in the README so each image states **what data** the viewer is seeing (e.g. portfolio learning note, a specific research run title).

## Evaluation screenshot

The Evaluation page expects a `report.json` from:

```bash
python scripts/run_retrieval_eval.py --help
```

Upload that file in the UI for a metrics table screenshot; it is independent of the portfolio seed.
