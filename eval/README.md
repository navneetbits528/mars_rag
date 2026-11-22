# Evaluation dataset

The benchmark is scoped to **NASA-HRP-EVIDENCE**. Do not add examples from the other NASA reports to this dataset.

## Ground-truth rules

- Every answerable question must have a concise reference answer and at least one exact `chunk_id` that supports it.
- Use `expected_pages` for the source PDF page numbers (1-based); do not guess page numbers.
- Every unanswerable question must use category `unanswerable`, set `answerable` to `false`, and leave `expected_answer` and `relevant_chunk_ids` empty.
- Categories: `factual`, `table_lookup`, `multi_hop`, `comparison`, `unanswerable`.
- Keep questions representative and avoid paraphrases that test the same fact repeatedly.
- A person must review every answer and evidence label against the PDF. Do not use model-generated answers as ground truth without review.

## Step 1: Export source chunks

After the baseline corpus is ingested and Postgres is running:

```powershell
uv run python -m marsrag.evaluation.dataset export-candidates --report-no NASA-HRP-EVIDENCE --output eval/candidates.jsonl
```

The JSONL contains chunk IDs, page numbers, section metadata, content type, and chunk text. Treat it as an annotation aid; it is not itself the evaluation set.

## Step 2: Curate the benchmark

Copy relevant chunk IDs and page numbers from `eval/candidates.jsonl` into `dataset.json`, following `dataset.template.json`. Add a balanced mix of factual, table lookup, multi-hop, comparison, and unanswerable questions. Start with approximately 20–30 carefully reviewed questions.

## Step 3: Validate

```powershell
uv run python -m marsrag.evaluation.dataset validate --input eval/dataset.json
uv run pytest tests/test_evaluation_dataset.py -v
```

The checked-in template intentionally contains zero examples. It is a schema template, not a usable benchmark; evaluation runs should require a non-empty, human-reviewed dataset.
