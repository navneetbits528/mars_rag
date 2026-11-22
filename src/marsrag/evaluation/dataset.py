from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from marsrag.db import fetch_all

EvalCategory = Literal[
    "factual",
    "table_lookup",
    "multi_hop",
    "comparison",
    "unanswerable",
]


class EvalExample(BaseModel):
    """One human-reviewed evaluation question with grounded evidence labels."""

    id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=5, max_length=2000)
    report_no: str = Field(min_length=1)
    category: EvalCategory
    answerable: bool
    expected_answer: str | None = None
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    expected_pages: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ground_truth(self) -> "EvalExample":
        if self.answerable:
            if not self.expected_answer or not self.expected_answer.strip():
                raise ValueError("Answerable examples require expected_answer.")
            if not self.relevant_chunk_ids:
                raise ValueError("Answerable examples require relevant_chunk_ids.")
            if self.category == "unanswerable":
                raise ValueError("Unanswerable category requires answerable=false.")
        else:
            if self.expected_answer is not None or self.relevant_chunk_ids:
                raise ValueError(
                    "Unanswerable examples must not contain an expected answer "
                    "or relevant chunk IDs."
                )
            if self.category != "unanswerable":
                raise ValueError("answerable=false requires category='unanswerable'.")
        if any(page < 1 for page in self.expected_pages):
            raise ValueError("expected_pages must contain positive page numbers.")
        if len(self.relevant_chunk_ids) != len(set(self.relevant_chunk_ids)):
            raise ValueError("relevant_chunk_ids must be unique.")
        return self


class EvalDataset(BaseModel):
    """Versioned, corpus-scoped evaluation set."""

    schema_version: Literal[1] = 1
    dataset_name: str = Field(min_length=1)
    corpus_report_no: str = Field(min_length=1)
    description: str = ""
    examples: list[EvalExample] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dataset(self) -> "EvalDataset":
        ids = [example.id for example in self.examples]
        if len(ids) != len(set(ids)):
            raise ValueError("Evaluation example IDs must be unique.")
        mismatched = [
            example.id for example in self.examples
            if example.report_no != self.corpus_report_no
        ]
        if mismatched:
            raise ValueError(
                "All examples must use corpus_report_no; mismatched IDs: "
                + ", ".join(mismatched)
            )
        return self


def load_dataset(path: str | Path) -> EvalDataset:
    """Load and validate a versioned JSON evaluation dataset."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalDataset.model_validate(payload)


def export_chunk_candidates(
    *,
    report_no: str,
    output_path: str | Path,
    limit: int = 1000,
) -> int:
    """Export current-corpus chunk evidence for human ground-truth annotation."""
    if not report_no or not report_no.strip():
        raise ValueError("report_no cannot be empty.")
    if not 1 <= limit <= 10000:
        raise ValueError("limit must be between 1 and 10000.")

    rows = fetch_all(
        """
        SELECT chunk_id, report_no, page, title, section_title, section_path,
               chunk_type, content
        FROM chunks
        WHERE report_no = %s AND status = 'current'
        ORDER BY page, chunk_id
        LIMIT %s
        """,
        (report_no, limit),
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mars RAG evaluation dataset tools.")
    commands = parser.add_subparsers(dest="command", required=True)

    validate_parser = commands.add_parser("validate", help="Validate a dataset JSON file.")
    validate_parser.add_argument("--input", required=True)

    export_parser = commands.add_parser(
        "export-candidates",
        help="Export current report chunks for human annotation.",
    )
    export_parser.add_argument("--report-no", required=True)
    export_parser.add_argument("--output", default="eval/candidates.jsonl")
    export_parser.add_argument("--limit", type=int, default=1000)

    args = parser.parse_args()
    if args.command == "validate":
        dataset = load_dataset(args.input)
        print(
            f"Valid dataset: {dataset.dataset_name}; "
            f"report={dataset.corpus_report_no}; examples={len(dataset.examples)}"
        )
    else:
        count = export_chunk_candidates(
            report_no=args.report_no,
            output_path=args.output,
            limit=args.limit,
        )
        print(f"Exported {count} chunk candidates to {args.output}")


if __name__ == "__main__":
    main()
