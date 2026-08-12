"""Validate golden-dataset structure and evidence provenance."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PAIR_FIELDS = {
    "id",
    "difficulty",
    "question",
    "expected_answer",
    "contexts",
    "attack_type",
}
DATASET_FIELDS = {"schema_version", "corpus_id", "qa_pairs"}
CONTEXT_FIELDS = {"source_doc", "text"}
SCHEMA_VERSION = "1.0"


class ConfigurationError(Exception):
    """The provided corpus or dataset cannot be read safely."""


@dataclass(frozen=True)
class Slot:
    id: str
    difficulty: str
    attack_type: str | None
    required_sources: frozenset[str]


@dataclass(frozen=True)
class Contract:
    schema_version: str
    corpus_id: str
    source_texts: dict[str, str]
    slots: tuple[Slot, ...]

    @property
    def difficulty_order(self) -> list[str]:
        return list(dict.fromkeys(slot.difficulty for slot in self.slots))


def _build_slots() -> tuple[Slot, ...]:
    """Return the fixed 5E + 7M + 5H + 3A dataset layout."""

    regular = [
        *(Slot(f"E{index:02d}", "easy", None, frozenset()) for index in range(1, 6)),
        *(
            Slot(f"M{index:02d}", "medium", None, frozenset())
            for index in range(1, 8)
        ),
        *(Slot(f"H{index:02d}", "hard", None, frozenset()) for index in range(1, 6)),
    ]
    scope_source = frozenset({"00_system_scope.md"})
    adversarial = [
        Slot("A01", "adversarial", "out_of_scope", scope_source),
        Slot("A02", "adversarial", "prompt_injection", scope_source),
        Slot(
            "A03",
            "adversarial",
            "false_premise_or_ambiguous_trap",
            scope_source,
        ),
    ]
    return tuple([*regular, *adversarial])


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"{label} is not valid JSON: {path}:{exc.lineno}:{exc.colno} ({exc.msg})"
        ) from exc
    except OSError as exc:
        raise ConfigurationError(
            f"Could not read {label.lower()} {path}: {exc}"
        ) from exc


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{location} must be a JSON object")
    return value


def _non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _required_text(item: dict[str, Any], field: str, location: str) -> str:
    value = item.get(field)
    if not _non_empty_text(value):
        raise ConfigurationError(f"{location}.{field} must be a non-empty string")
    return value.strip()


def _safe_source_path(corpus_root: Path, source_doc: str) -> Path:
    relative = Path(source_doc)
    if relative.is_absolute():
        raise ConfigurationError(f"Source path must be relative: {source_doc}")
    path = (corpus_root / relative).resolve()
    if not path.is_relative_to(corpus_root):
        raise ConfigurationError(f"Source path escapes corpus directory: {source_doc}")
    if not path.is_file():
        raise ConfigurationError(f"Source document not found: {path}")
    return path


def build_contract(corpus_dir: Path) -> Contract:
    """Build the validation contract from the corpus manifest."""

    corpus_root = corpus_dir.resolve()
    manifest = _object(
        _read_json(corpus_root / "manifest.json", "Manifest"), "Manifest"
    )
    corpus_id = _required_text(manifest, "corpus_id", "Manifest")
    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ConfigurationError("Manifest.documents must be a non-empty list")

    source_texts: dict[str, str] = {}
    for index, document in enumerate(documents, start=1):
        item = _object(document, f"Manifest.documents[{index}]")
        source_doc = _required_text(item, "path", f"Manifest.documents[{index}]")
        if source_doc in source_texts:
            raise ConfigurationError(f"Duplicate manifest source path: {source_doc}")
        path = _safe_source_path(corpus_root, source_doc)
        try:
            source_texts[source_doc] = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigurationError(
                f"Could not read source document {path}: {exc}"
            ) from exc

    return Contract(SCHEMA_VERSION, corpus_id, source_texts, _build_slots())


def _normalized_question(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip().casefold()


def _validate_contexts(
    pair: dict[str, Any],
    label: str,
    contract: Contract,
    errors: list[str],
) -> set[str]:
    contexts = pair["contexts"]
    if not isinstance(contexts, list) or not contexts:
        errors.append(f"{label}: contexts must be a non-empty list")
        return set()

    used_sources: set[str] = set()
    seen_evidence: set[tuple[str, str]] = set()
    for index, context in enumerate(contexts, start=1):
        location = f"{label} contexts[{index}]"
        if not isinstance(context, dict):
            errors.append(f"{location} must be an object")
            continue

        missing_fields = sorted(CONTEXT_FIELDS - context.keys())
        unexpected_fields = sorted(context.keys() - CONTEXT_FIELDS)
        if missing_fields:
            errors.append(f"{location}: missing fields: {', '.join(missing_fields)}")
            continue
        if unexpected_fields:
            errors.append(
                f"{location}: unexpected fields: {', '.join(unexpected_fields)}"
            )

        source_doc = context.get("source_doc")
        evidence = context.get("text")
        if not _non_empty_text(source_doc):
            errors.append(f"{location}.source_doc must be a non-empty string")
            continue
        if not _non_empty_text(evidence):
            errors.append(f"{location}.text must be a non-empty string")
            continue
        source_doc, evidence = source_doc.strip(), evidence.strip()

        if source_doc not in contract.source_texts:
            errors.append(f"{location}: source_doc {source_doc!r} is not in manifest")
            continue
        key = (source_doc, evidence)
        if key in seen_evidence:
            errors.append(f"{location}: duplicate evidence within {label}")
        seen_evidence.add(key)
        if evidence not in contract.source_texts[source_doc]:
            errors.append(
                f"{location}: text is not a verbatim substring of {source_doc}"
            )
            continue
        used_sources.add(source_doc)
    return used_sources


def validate_dataset(
    dataset: Any,
    contract: Contract,
) -> tuple[list[str], dict[str, Any]]:
    """Return all deterministic student-data errors and summary statistics."""

    errors: list[str] = []
    stats: dict[str, Any] = {
        "qa_count": 0,
        "difficulty_counts": Counter(),
        "used_documents": set(),
    }
    if not isinstance(dataset, dict):
        return ["Dataset root must be a JSON object"], stats

    missing_root_fields = sorted(DATASET_FIELDS - dataset.keys())
    unexpected_root_fields = sorted(dataset.keys() - DATASET_FIELDS)
    if missing_root_fields:
        errors.append("Dataset is missing fields: " + ", ".join(missing_root_fields))
    if unexpected_root_fields:
        errors.append(
            "Dataset has unexpected fields: " + ", ".join(unexpected_root_fields)
        )

    if dataset.get("schema_version") != contract.schema_version:
        errors.append(
            f"schema_version must be {contract.schema_version!r}; "
            f"found {dataset.get('schema_version')!r}"
        )
    if dataset.get("corpus_id") != contract.corpus_id:
        errors.append(
            f"corpus_id must be {contract.corpus_id!r}; "
            f"found {dataset.get('corpus_id')!r}"
        )

    pairs = dataset.get("qa_pairs")
    if not isinstance(pairs, list):
        return [*errors, "qa_pairs must be a list"], stats
    stats["qa_count"] = len(pairs)
    if len(pairs) != len(contract.slots):
        errors.append(
            f"qa_pairs must contain exactly {len(contract.slots)} records; "
            f"found {len(pairs)}"
        )

    seen_ids: set[str] = set()
    seen_questions: dict[str, str] = {}
    for index, pair in enumerate(pairs):
        position = index + 1
        fallback = f"qa_pairs[{position}]"
        if not isinstance(pair, dict):
            errors.append(f"{fallback} must be an object")
            continue

        pair_id = pair.get("id")
        label = str(pair_id) if _non_empty_text(pair_id) else fallback
        missing = sorted(PAIR_FIELDS - pair.keys())
        unexpected = sorted(pair.keys() - PAIR_FIELDS)
        if missing:
            errors.append(f"{label}: missing fields: {', '.join(missing)}")
            continue
        if unexpected:
            errors.append(f"{label}: unexpected fields: {', '.join(unexpected)}")

        if not _non_empty_text(pair_id):
            errors.append(f"{fallback}.id must be a non-empty string")
        elif pair_id in seen_ids:
            errors.append(f"Duplicate QA id: {pair_id}")
        else:
            seen_ids.add(pair_id)

        slot = contract.slots[index] if index < len(contract.slots) else None
        if slot is not None:
            if pair_id != slot.id:
                errors.append(
                    f"Position {position}: expected id {slot.id}, found {pair_id!r}"
                )
            if pair["difficulty"] != slot.difficulty:
                errors.append(
                    f"{label}: expected difficulty {slot.difficulty!r}, "
                    f"found {pair['difficulty']!r}"
                )
            if pair["attack_type"] != slot.attack_type:
                errors.append(
                    f"{label}: expected attack_type {slot.attack_type!r}, "
                    f"found {pair['attack_type']!r}"
                )

        difficulty = pair["difficulty"]
        if isinstance(difficulty, str):
            stats["difficulty_counts"][difficulty] += 1

        question = pair["question"]
        if not _non_empty_text(question):
            errors.append(f"{label}: question must be a non-empty string")
        else:
            normalized = _normalized_question(question)
            if normalized in seen_questions:
                errors.append(
                    f"{label}: question duplicates {seen_questions[normalized]} exactly"
                )
            else:
                seen_questions[normalized] = label

        if not _non_empty_text(pair["expected_answer"]):
            errors.append(f"{label}: expected_answer must be a non-empty string")

        pair_sources = _validate_contexts(pair, label, contract, errors)
        stats["used_documents"].update(pair_sources)
        if slot is not None:
            missing_sources = sorted(slot.required_sources - pair_sources)
            if missing_sources:
                errors.append(
                    f"{label}: missing source documents required by dataset contract: "
                    + ", ".join(missing_sources)
                )

    missing_documents = sorted(set(contract.source_texts) - stats["used_documents"])
    if missing_documents:
        errors.append(
            "Dataset must use every source document at least once; missing: "
            + ", ".join(missing_documents)
        )
    return errors, stats


def _print_results(
    dataset_path: Path,
    corpus_dir: Path,
    contract: Contract,
    errors: list[str],
    stats: dict[str, Any],
) -> None:
    counts: Counter[str] = stats["difficulty_counts"]
    summary = ", ".join(
        f"{difficulty}={counts[difficulty]}" for difficulty in contract.difficulty_order
    )
    print(f"Dataset: {dataset_path}")
    print(f"Corpus:  {corpus_dir}")
    print(f"QA pairs: {stats['qa_count']}")
    print(f"Difficulty: {summary or 'none'}")
    print(
        f"Document coverage: {len(stats['used_documents'])}/"
        f"{len(contract.source_texts)}"
    )

    if errors:
        print(f"\nFAILED ({len(errors)} errors):")
        for error in errors:
            print(f"  - {error}")
        return
    print("\nPASS: dataset structure and evidence provenance are valid.")
    print("Note: semantic quality and true difficulty still require rubric review.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("golden_dataset.json"),
        help="Dataset to validate (default: golden_dataset.json)",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/student_services"),
        help="Corpus directory (default: data/student_services)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    corpus_dir = args.corpus_dir.resolve()
    try:
        contract = build_contract(corpus_dir)
        dataset = _read_json(dataset_path, "Dataset")
    except ConfigurationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    errors, stats = validate_dataset(dataset, contract)
    _print_results(dataset_path, corpus_dir, contract, errors, stats)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
