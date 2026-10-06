# Evaluation datasets

## Format (`version` 1.0)

Each dataset is a JSON file:

| Field | Description |
| --- | --- |
| `question` | Natural-language query |
| `expected_sources` | Optional document-level labels (`document_id`, `filename`, or `external_id`) |
| `expected_evidence` | **chunk_id** list used for Recall@K, Precision@K, MRR, and nDCG |
| `reference_answer` | Gold answer for answer-quality metrics |
| `retrieval_sub_queries` | Optional sub-questions for **agentic** retrieval |

## Labeling chunk IDs

From the repo root (with a processed corpus under `data/processed/`):

```bash
cd backend
python ../scripts/list_eval_chunk_ids.py --query "hybrid retrieval" --top-k 10
```

Copy relevant `chunk_id` values into `expected_evidence` for each item.

## Running evaluation

```bash
cd backend
python ../scripts/run_retrieval_eval.py \
  --dataset ../data/evaluation/sample_dataset.json \
  --output ../data/evaluation/reports/$(date +%Y%m%d_%H%M%S) \
  --k 1 3 5 10 \
  --seed 42
```

Outputs:

- `report.json` — full metrics per strategy and per query
- `report.md` — leaderboard and summary
- `config.json` — settings snapshot for reproducibility

Use `--evaluate-answers` to score faithfulness, relevance, and citation metrics (requires a configured LLM).
