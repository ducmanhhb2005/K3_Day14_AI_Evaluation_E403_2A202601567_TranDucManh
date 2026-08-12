"""Measure the optional overlap reranker on real saved retrieval traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from template import RAGASEvaluator, rerank_by_overlap


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def measure_reranking(
    golden_path: Path,
    actual_path: Path,
    limit: int = 5,
) -> dict[str, Any]:
    golden = _read(golden_path)
    actual = _read(actual_path)
    gold_by_id = {item["id"]: item for item in golden["qa_pairs"]}
    answer_by_id = {item["id"]: item for item in actual["answers"]}
    evaluator = RAGASEvaluator()
    candidates: list[dict[str, Any]] = []

    for item_id, gold in gold_by_id.items():
        record = answer_by_id[item_id]
        contexts = [chunk["text"] for chunk in record["retrieved_contexts"]]
        expected = gold["expected_answer"]
        before_recall = evaluator.evaluate_context_recall(contexts, expected)
        before_precision = evaluator.evaluate_context_precision(contexts, expected)
        reranked = rerank_by_overlap(contexts, expected)
        after_recall = evaluator.evaluate_context_recall(reranked, expected)
        after_precision = evaluator.evaluate_context_precision(reranked, expected)
        candidates.append(
            {
                "id": item_id,
                "recall_before": before_recall,
                "recall_after": after_recall,
                "precision_before": before_precision,
                "precision_after": after_precision,
                "delta_precision": after_precision - before_precision,
                "chunk_count": len(contexts),
            }
        )

    selected = sorted(
        candidates,
        key=lambda row: (row["precision_before"], -row["delta_precision"], row["id"]),
    )[:limit]
    return {
        "method": "lexical overlap reranking on the unchanged retrieved set",
        "case_count": len(selected),
        "results": selected,
        "averages": {
            key: sum(row[key] for row in selected) / len(selected)
            for key in (
                "recall_before",
                "recall_after",
                "precision_before",
                "precision_after",
                "delta_precision",
            )
        },
    }


def markdown_table(report: dict[str, Any]) -> str:
    lines = [
        "| ID | Recall before | Recall after | Precision before | Precision after | Delta Precision |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['id']} | {row['recall_before']:.3f} | {row['recall_after']:.3f} | "
            f"{row['precision_before']:.3f} | {row['precision_after']:.3f} | "
            f"{row['delta_precision']:+.3f} |"
        )
    avg = report["averages"]
    lines.append(
        f"| **Avg** | {avg['recall_before']:.3f} | {avg['recall_after']:.3f} | "
        f"{avg['precision_before']:.3f} | {avg['precision_after']:.3f} | "
        f"{avg['delta_precision']:+.3f} |"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=Path("golden_dataset.json"))
    parser.add_argument("--actual", type=Path, default=Path("artifacts/actual_answers.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/bonus_reranking.json"))
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    try:
        report = measure_reranking(args.golden, args.actual, args.limit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(markdown_table(report))
    print(f"\nSaved: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
