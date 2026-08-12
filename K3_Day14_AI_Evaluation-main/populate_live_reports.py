"""Populate Exercise 3.2, Exercise 3.5 and reflection from real artifacts.

Run only after ``domain_assistant.py`` and ``evaluate_answers.py`` (or the UI)
have completed. The script refuses to proceed if the artifacts are incomplete.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from bonus_reranking import markdown_table, measure_reranking


ROOT = Path(__file__).resolve().parent


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_artifacts(
    golden: dict[str, Any], actual: dict[str, Any], benchmark: dict[str, Any]
) -> None:
    expected_ids = [item["id"] for item in golden.get("qa_pairs", [])]
    actual_ids = [item["id"] for item in actual.get("answers", [])]
    result_ids = [item["id"] for item in benchmark.get("results", [])]
    if len(expected_ids) != 20 or actual_ids != expected_ids or result_ids != expected_ids:
        raise ValueError("Artifacts must contain the same ordered set of 20 golden IDs")
    if any(item.get("error") is not None or not item.get("actual_answer") for item in actual["answers"]):
        raise ValueError("Actual-answer artifact contains an error or empty answer")
    if benchmark.get("summary", {}).get("total") != 20:
        raise ValueError("Benchmark summary total must be 20")


def short_question(question: str, limit: int = 47) -> str:
    text = re.sub(r"\s+", " ", question).replace("|", "\\|")
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def build_exercise_3_2(benchmark: dict[str, Any]) -> str:
    summary = benchmark["summary"]
    results = benchmark["results"]
    lines = [
        "### Exercise 3.2 — Benchmark Run",
        "",
        "> **Kết quả live:** Bảng dưới được tạo từ `artifacts/actual_answers.json`",
        "> và `artifacts/benchmark_results.json`; không dùng sample hoặc placeholder.",
        "",
        "| ID | Question (short) | Ctx Recall | Ctx Precision | Faithfulness | Relevance | Completeness | Overall | Passed? | Failure Type |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['id']} | {short_question(row['question'])} | "
            f"{row['context_recall']:.3f} | {row['context_precision']:.3f} | "
            f"{row['faithfulness']:.3f} | {row['relevance']:.3f} | "
            f"{row['completeness']:.3f} | {row['overall']:.3f} | "
            f"{'Yes' if row['passed'] else 'No'} | {row['failure_type'] or '-'} |"
        )
    lines.extend(
        [
            "",
            "**Aggregate Report**",
            "",
            f"- Overall pass rate: {summary['pass_rate']:.1%}",
            f"- Avg Context Recall: {summary['avg_context_recall']:.3f}",
            f"- Avg Context Precision: {summary['avg_context_precision']:.3f}",
            f"- Avg Faithfulness: {summary['avg_faithfulness']:.3f}",
            f"- Avg Relevance: {summary['avg_relevance']:.3f}",
            f"- Avg Completeness: {summary['avg_completeness']:.3f}",
            f"- Failure type distribution: {summary['failure_types']}",
            "",
            "**Ba cases có Overall Score thấp nhất**",
            "",
        ]
    )
    worst = sorted(results, key=lambda row: row["overall"])[:3]
    for index, row in enumerate(worst, start=1):
        lines.append(
            f"{index}. ID: {row['id']} | Score: {row['overall']:.3f} | "
            f"Failure type: {row['failure_type'] or '-'}"
        )
    metric_averages = {
        "Context Recall": summary["avg_context_recall"],
        "Context Precision": summary["avg_context_precision"],
        "Faithfulness": summary["avg_faithfulness"],
        "Relevance": summary["avg_relevance"],
        "Completeness": summary["avg_completeness"],
    }
    weakest = min(metric_averages, key=metric_averages.get)
    retrieval_avg = (summary["avg_context_recall"] + summary["avg_context_precision"]) / 2
    answer_avg = (
        summary["avg_faithfulness"] + summary["avg_relevance"] + summary["avg_completeness"]
    ) / 3
    locus = "retrieval" if retrieval_avg + 0.03 < answer_avg else "generation/answer quality" if answer_avg + 0.03 < retrieval_avg else "cả retrieval và generation"
    lines.extend(
        [
            "",
            "**Nhận xét ngắn:** Metric nào yếu nhất? Kết quả gợi ý vấn đề nằm ở retrieval hay generation?",
            "",
            f"> Metric trung bình yếu nhất là **{weakest}** ({metric_averages[weakest]:.3f}). "
            f"Retrieval average là {retrieval_avg:.3f}, answer-side average là {answer_avg:.3f}; "
            f"vì vậy dấu hiệu chính nằm ở **{locus}**. Kết luận này cần được kiểm tra lại bằng trace của ba case thấp nhất.",
            "",
        ]
    )
    return "\n".join(lines)


def root_analysis(row: dict[str, Any], actual: dict[str, Any], gold: dict[str, Any]) -> dict[str, str]:
    retrieved_sources = [chunk["source_doc"] for chunk in actual["retrieved_contexts"]]
    gold_sources = [context["source_doc"] for context in gold["contexts"]]
    missing = sorted(set(gold_sources) - set(retrieved_sources))
    evidence = (
        f"Gold sources: {', '.join(gold_sources)}. Retrieved sources: {', '.join(retrieved_sources)}. "
        + (f"Missing gold source(s): {', '.join(missing)}." if missing else "All gold source names appear in the retrieved trace; inspect paragraph coverage and ranking.")
    )
    scores = {key: row[key] for key in ("faithfulness", "relevance", "completeness")}
    weakest = min(scores, key=scores.get)
    recall = row["context_recall"]
    if recall < 0.6:
        symptom = "The answer misses or distorts required evidence and retrieval recall is low."
        why1 = "The retrieved set does not lexically cover enough of the expected answer."
        why2 = "BM25 ranked chunks from overlapping terms but missed part of the multi-condition evidence."
        why3 = "The query and corpus use different vocabulary or required evidence is split across paragraphs/documents."
        why4 = "The baseline has no semantic query expansion, metadata/date filter, or cross-encoder retrieval stage."
        root = "Retrieval is lexical-only and not optimized for multi-document policy conditions."
        fix = "Add query expansion/metadata-aware retrieval and verify Context Recall on this case."
    elif weakest == "completeness":
        symptom = "Retrieved evidence is available, but the answer omits required conditions or exceptions."
        why1 = "The generator selected only part of the retrieved evidence."
        why2 = "The response prompt requests concision and does not enforce a per-policy requirement checklist."
        why3 = "No structured answer plan checks dates, amounts, conditions and exceptions before finalization."
        why4 = "The pipeline evaluates completeness after generation but has no pre-output completeness guard."
        root = "Generation lacks a structured coverage check for multi-condition answers."
        fix = "Add a required-claims checklist and verify Completeness without reducing Faithfulness."
    elif weakest == "faithfulness":
        symptom = "The answer contains tokens or claims not sufficiently grounded in gold evidence."
        why1 = "The generator added wording beyond the supported context."
        why2 = "The baseline prompt asks for a natural answer but does not validate each claim against evidence."
        why3 = "There is no claim extraction and entailment/grounding pass before returning the response."
        why4 = "Faithfulness is measured offline only, not enforced as an output guard."
        root = "The generator has no claim-level grounding gate."
        fix = "Add claim-level evidence verification and verify Faithfulness on this trace."
    else:
        symptom = "The answer does not sufficiently address the user's exact intent."
        why1 = "The response focuses on related context rather than every part of the question."
        why2 = "Intent decomposition is implicit and can be lost in a multi-part or adversarial query."
        why3 = "The prompt has no explicit question-part checklist."
        why4 = "The pipeline has no intent-coverage check before final output."
        root = "Generation lacks explicit intent decomposition and coverage validation."
        fix = "Add intent decomposition and verify Relevance on this case."
    return {
        "evidence": evidence,
        "symptom": symptom,
        "why1": why1,
        "why2": why2,
        "why3": why3,
        "why4": why4,
        "root": root,
        "fix": fix,
    }


def build_reflection(
    golden: dict[str, Any], actual: dict[str, Any], benchmark: dict[str, Any]
) -> str:
    results = benchmark["results"]
    summary = benchmark["summary"]
    answer_by_id = {item["id"]: item for item in actual["answers"]}
    gold_by_id = {item["id"]: item for item in golden["qa_pairs"]}
    metric_keys = [
        ("Context Recall", "context_recall"),
        ("Context Precision", "context_precision"),
        ("Faithfulness", "faithfulness"),
        ("Relevance", "relevance"),
        ("Completeness", "completeness"),
        ("Overall Score", "overall"),
    ]
    lines = [
        "# Day 14 — Reflection",
        "",
        "## Evaluation Report & Failure Analysis",
        "",
        "> Generated only from the completed real OpenAI and benchmark artifacts.",
        "",
        "## 1. Benchmark Results Summary",
        "",
        f"**Overall pass rate:** {summary['pass_rate']:.1%}",
        "",
        "| Metric | Average | Min | Max | Nhận xét |",
        "|---|---:|---:|---:|---|",
    ]
    averages: dict[str, float] = {}
    for label, key in metric_keys:
        values = [float(row[key]) for row in results]
        average = sum(values) / len(values)
        averages[label] = average
        interpretation = "Good" if average >= 0.8 else "Needs Work" if average >= 0.6 else "Significant Issues"
        lines.append(f"| {label} | {average:.3f} | {min(values):.3f} | {max(values):.3f} | {interpretation} |")
    bands = {
        "Good (0.8–1.0)": [name for name, value in averages.items() if value >= 0.8],
        "Needs Work (0.6–0.8)": [name for name, value in averages.items() if 0.6 <= value < 0.8],
        "Significant Issues (<0.6)": [name for name, value in averages.items() if value < 0.6],
    }
    lines.extend(["", "**Score interpretation**", ""])
    for band, names in bands.items():
        lines.append(f"- {band}: {', '.join(names) if names else 'None'}")
    failures = [row for row in results if not row["passed"]]
    counts = Counter(row["failure_type"] or "unknown" for row in failures)
    lines.extend(["", "**Failure type distribution**", "", "| Failure Type | Count | Percentage |", "|---|---:|---:|"])
    for failure_type in ("hallucination", "irrelevant", "incomplete", "off_topic", "refusal"):
        count = counts[failure_type]
        percentage = count / len(results)
        lines.append(f"| {failure_type} | {count} | {percentage:.1%} |")
    retrieval_avg = (averages["Context Recall"] + averages["Context Precision"]) / 2
    answer_avg = (averages["Faithfulness"] + averages["Relevance"] + averages["Completeness"]) / 3
    lines.extend(
        [
            "",
            "**Chẩn đoán tổng quan:**",
            "",
            f"> Retrieval average là {retrieval_avg:.3f}; answer-side average là {answer_avg:.3f}. "
            "Các trace thấp nhất bên dưới được dùng để phân biệt retrieval miss với generation miss; không kết luận chỉ từ pass rate.",
            "",
            "## 2. Top 3 Worst Failures — 5 Whys",
            "",
        ]
    )
    worst = sorted(results, key=lambda row: row["overall"])[:3]
    analyses: list[tuple[dict[str, Any], dict[str, str]]] = []
    for index, row in enumerate(worst, start=1):
        gold = gold_by_id[row["id"]]
        answer = answer_by_id[row["id"]]
        analysis = root_analysis(row, answer, gold)
        analyses.append((row, analysis))
        lines.extend(
            [
                f"### Failure {index} — {row['id']}",
                "",
                f"**Question:** {row['question']}",
                "",
                f"**Expected answer:** {gold['expected_answer']}",
                "",
                f"**Actual answer:** {answer['actual_answer']}",
                "",
                f"**Scores:** Context Recall {row['context_recall']:.3f} | Context Precision {row['context_precision']:.3f} | Faithfulness {row['faithfulness']:.3f} | Relevance {row['relevance']:.3f} | Completeness {row['completeness']:.3f} | Overall {row['overall']:.3f}",
                "",
                f"**Evidence inspection:** {analysis['evidence']}",
                "",
                "| Level | Question | Answer |",
                "|---|---|---|",
                f"| Symptom | Vấn đề quan sát được là gì? | {analysis['symptom']} |",
                f"| Why 1 | Tại sao symptom xảy ra? | {analysis['why1']} |",
                f"| Why 2 | Tại sao nguyên nhân trên xảy ra? | {analysis['why2']} |",
                f"| Why 3 | Tại sao vấn đề đó chưa được ngăn chặn? | {analysis['why3']} |",
                f"| Why 4 | Tại sao cơ chế hiện tại chưa phát hiện/xử lý? | {analysis['why4']} |",
                f"| Why 5 | Root cause có thể hành động được là gì? | {analysis['root']} |",
                "",
                f"**Proposed fix:** {analysis['fix']}",
                "",
            ]
        )
    cluster_map: dict[str, list[str]] = {}
    for row, analysis in analyses:
        cluster_map.setdefault(analysis["root"], []).append(row["id"])
    lines.extend(["## 3. Failure Clustering", "", "| Cluster | Root Cause | Failure IDs | Priority |", "|---:|---|---|---|"])
    for index, (root, ids) in enumerate(cluster_map.items(), start=1):
        lines.append(f"| {index} | {root} | {', '.join(ids)} | {'High' if index == 1 else 'Medium'} |")
    first_root = next(iter(cluster_map))
    lines.extend(
        [
            "",
            f"> Nếu chỉ sửa một cluster, ưu tiên **{first_root}** vì nó xuất hiện trong nhóm case có overall thấp nhất; verify bằng metric gắn trực tiếp với proposed fix và chạy regression trên đủ 20 cases.",
            "",
            "## 4. Improvement Log",
            "",
            benchmark["failure_analysis"].get("improvement_log", "No failures."),
            "",
            "**Ba improvement suggestions ưu tiên**",
            "",
        ]
    )
    suggestions = benchmark["failure_analysis"].get("suggestions", [])[:3]
    for index, suggestion in enumerate(suggestions, start=1):
        lines.append(f"{index}. {suggestion}")
    lines.extend(
        [
            "",
            "| Suggestion | Target metric | Verification method |",
            "|---|---|---|",
        ]
    )
    targets = ["Faithfulness", "Context Recall / Context Precision", "Completeness / Relevance"]
    for suggestion, target in zip(suggestions, targets):
        lines.append(f"| {suggestion} | {target} | Re-run the same 20-case benchmark and `run_regression()`; inspect top-3 traces. |")
    lines.extend(
        [
            "",
            "## 5. Regression Testing Strategy",
            "",
            "**Câu 1:** Chạy `run_regression()` ở mọi pull request thay đổi prompt, model, retrieval, chunking hoặc guardrail; chạy lại trước release và theo lịch khi corpus/model thay đổi.",
            "",
            "**Câu 2:** Drop 0.05 phù hợp làm regression alarm ban đầu, nhưng không đủ cho case safety-critical. Một privacy leak, prompt-injection success hoặc material hallucination phải block ngay dù aggregate drop nhỏ hơn 0.05.",
            "",
            "**Câu 3:** Block nếu Faithfulness aggregate <0.70, bất kỳ hard/adversarial case core score <0.50, hoặc có safety/privacy failure. Context Precision thấp nhưng Recall và answer metrics ổn có thể alert để tối ưu thay vì block.",
            "",
            "```text",
            "Code/prompt/retrieval change → Offline golden benchmark → Regression gate → Human review for critical failures → Deploy",
            "```",
            "",
            "## 6. Continuous Improvement Loop",
            "",
            "| Priority | Action | Metric dự kiến cải thiện | Expected impact |",
            "|---:|---|---|---|",
        ]
    )
    for index, suggestion in enumerate(suggestions, start=1):
        lines.append(f"| {index} | {suggestion} | {targets[index - 1]} | Lift the affected failure cluster without regression on the remaining cases. |")
    lines.extend(
        [
            "",
            f"**Cases cần giữ/thêm ở vòng tiếp theo:** {', '.join(row['id'] for row in worst)} plus paraphrased variants that preserve the same policy rule but change vocabulary.",
            "",
            "## 7. Final Reflection",
            "",
            f"> Kết quả đáng chú ý nhất là chênh lệch giữa retrieval average ({retrieval_avg:.3f}) và answer-side average ({answer_avg:.3f}); trace cho thấy score thấp không tự động đồng nghĩa cùng một root cause.",
            "",
            "> Word-overlap bỏ qua synonym, paraphrase, negation, entailment và mức độ quan trọng của từng claim; đồng thời có thể thưởng việc lặp từ khóa. Production nên bổ sung semantic/claim-level groundedness, calibrated LLM judge với human labels, task/safety assertions và online monitoring cho latency, cost, drift, feedback.",
            "",
        ]
    )
    return "\n".join(lines)


def replace_section(text: str, start: str, end: str, replacement: str) -> str:
    pattern = re.compile(rf"{re.escape(start)}.*?(?={re.escape(end)})", re.S)
    updated, count = pattern.subn(replacement.rstrip() + "\n\n", text, count=1)
    if count != 1:
        raise ValueError(f"Could not locate section {start!r}")
    return updated


def main() -> int:
    try:
        golden_path = ROOT / "golden_dataset.json"
        actual_path = ROOT / "artifacts" / "actual_answers.json"
        benchmark_path = ROOT / "artifacts" / "benchmark_results.json"
        golden = read_json(golden_path)
        actual = read_json(actual_path)
        benchmark = read_json(benchmark_path)
        validate_artifacts(golden, actual, benchmark)

        exercises_path = ROOT / "exercises.md"
        exercises = exercises_path.read_text(encoding="utf-8")
        exercises = replace_section(
            exercises,
            "### Exercise 3.2 — Benchmark Run",
            "### Exercise 3.3 — LLM-as-a-Judge Rubric Design",
            build_exercise_3_2(benchmark),
        )

        reranking = measure_reranking(golden_path, actual_path, 5)
        old_table = re.compile(
            r"\| ID \| Recall before \| Recall after \| Precision before \| Precision after \| Delta Precision \|.*?\| \*\*Avg\*\* \|.*?\|",
            re.S,
        )
        exercises, count = old_table.subn(markdown_table(reranking), exercises, count=1)
        if count != 1:
            raise ValueError("Could not locate Exercise 3.5 result table")
        exercises = exercises.replace(
            "- [ ] Exercise 3.2 có năm metrics, aggregate report và ba cases thấp nhất.",
            "- [x] Exercise 3.2 có năm metrics, aggregate report và ba cases thấp nhất.",
        ).replace(
            "- [ ] `reflection.md` có ba failure analyses và regression strategy.",
            "- [x] `reflection.md` có ba failure analyses và regression strategy.",
        )
        exercises_path.write_text(exercises, encoding="utf-8")
        (ROOT / "reflection.md").write_text(
            build_reflection(golden, actual, benchmark), encoding="utf-8"
        )
        reranking_path = ROOT / "artifacts" / "bonus_reranking.json"
        reranking_path.write_text(json.dumps(reranking, indent=2) + "\n", encoding="utf-8")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        print("No report was populated from incomplete or invalid artifacts.")
        return 2
    print("Updated exercises.md and reflection.md from verified real artifacts.")
    print(f"Saved bonus reranking report: {reranking_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
