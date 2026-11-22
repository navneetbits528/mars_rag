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

## Step 4: Evaluate semantic cache thresholds

Create `eval/cache_dataset.json` from `cache_dataset.template.json`. Each pair must be reviewed by a person:
- `equivalent=true`: reusing the cached answer for the candidate question is safe.
- `equivalent=false`: the candidate asks for materially different information, so reuse is unsafe.

Include paraphrases, close-but-different questions, different numeric parameters, and questions about different mission phases. Do not label pairs equivalent merely because they share keywords.

Run:

```powershell
uv run python -m marsrag.evaluation.cache --input eval/cache_dataset.json --output eval/cache_results.json
```

The runner embeds each unique question once, then compares cosine similarity at 0.90, 0.93, 0.95, and 0.97. It reports hit rate, precision, recall, false-positive rate, and per-pair decisions.

A false positive is especially important: it means the threshold would reuse an answer for a question labelled non-equivalent. Prefer a threshold with a low false-positive rate, while retaining useful recall.

This experiment evaluates semantic-match decisions, not end-to-end PostgreSQL lookup latency or answer correctness after retrieval. Those require a separate live-cache benchmark. Keep the human-labelled pair set balanced and review it against the source report.

