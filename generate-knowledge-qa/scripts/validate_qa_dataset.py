#!/usr/bin/env python3
"""Validate the final GraphRAG local-query QA dataset."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence


TOP_LEVEL_KEYS = ["dataset_name", "source_file", "design_principles", "cases"]
CASE_KEYS = [
    "query_id",
    "category",
    "question",
    "groundtruth_knowledge_ids",
    "groundtruth_answer",
    "design_intent",
]
META_TERMS = ("knowledge_id", "原子知识", "知识库", "根据文本", "根据材料")
CATEGORY_RE = re.compile(r"^[\u4e00-\u9fff]{2,6}-[\u4e00-\u9fff]{2,6}$")
LIST_ITEM_RE = re.compile(r"(?m)^\s*(?:[-*•]\s+|\d+[.)、](?!\d)|[一二三四五六七八九十]+[、.)])")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def add_error(errors: list[str], path: str, message: str) -> None:
    errors.append(f"{path}: {message}")


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_dataset(data: Any, known_ids: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["$: dataset must be a JSON object"]

    for key in TOP_LEVEL_KEYS:
        if key not in data:
            add_error(errors, key, "missing required top-level field")
    if set(data) != set(TOP_LEVEL_KEYS):
        add_error(errors, "$", f"fields must be exactly {TOP_LEVEL_KEYS}")
    for key in ("dataset_name", "source_file"):
        if not nonempty_string(data.get(key)):
            add_error(errors, key, "must be a non-empty string")
    principles = data.get("design_principles")
    if not isinstance(principles, list) or not principles or not all(nonempty_string(item) for item in principles):
        add_error(errors, "design_principles", "must be a non-empty list of non-empty strings")

    cases = data.get("cases")
    if not isinstance(cases, list):
        add_error(errors, "cases", "must be a list")
        return errors
    if not cases:
        add_error(errors, "cases", "must contain at least one final case")

    seen_questions: set[str] = set()
    for index, case in enumerate(cases):
        path = f"cases[{index}]"
        if not isinstance(case, dict):
            add_error(errors, path, "must be an object")
            continue
        if set(case) != set(CASE_KEYS):
            add_error(errors, path, f"fields must be exactly {CASE_KEYS}")

        expected_query_id = f"Q{index + 1:03d}"
        if case.get("query_id") != expected_query_id:
            add_error(errors, f"{path}.query_id", f"must be {expected_query_id}")

        ids = case.get("groundtruth_knowledge_ids")
        if not isinstance(ids, list):
            add_error(errors, f"{path}.groundtruth_knowledge_ids", "must be a list")
        else:
            if len(ids) < 3:
                add_error(errors, f"{path}.groundtruth_knowledge_ids", "must contain at least 3 ids")
            valid_id_types = all(nonempty_string(kid) and kid == kid.strip() for kid in ids)
            if not valid_id_types:
                add_error(errors, f"{path}.groundtruth_knowledge_ids", "ids must be non-empty strings without outer whitespace")
            elif len(ids) != len(set(ids)):
                add_error(errors, f"{path}.groundtruth_knowledge_ids", "must not contain duplicates")
            if valid_id_types and known_ids is not None:
                unknown = sorted(set(ids) - known_ids)
                if unknown:
                    add_error(errors, f"{path}.groundtruth_knowledge_ids", f"unknown source ids: {unknown}")

        question = case.get("question")
        if not isinstance(question, str):
            add_error(errors, f"{path}.question", "must be a string")
        else:
            if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", question):
                add_error(errors, f"{path}.question", "must contain a meaningful question")
            if question.count("？") != 1:
                add_error(errors, f"{path}.question", "must contain exactly one Chinese question mark")
            if "?" in question:
                add_error(errors, f"{path}.question", "must not contain half-width question mark")
            for term in META_TERMS:
                if term in question:
                    add_error(errors, f"{path}.question", f"must not contain metadata term {term}")
            if isinstance(ids, list):
                for kid in ids:
                    if isinstance(kid, str) and re.fullmatch(r"[A-Za-z0-9_-]+", kid) and re.search(r"[A-Za-z]", kid):
                        if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(kid)}(?![A-Za-z0-9_-])", question):
                            add_error(errors, f"{path}.question", f"must not expose knowledge id {kid}")
            normalized_question = re.sub(r"\s+", "", question)
            if normalized_question in seen_questions:
                add_error(errors, f"{path}.question", "duplicate question")
            seen_questions.add(normalized_question)

        category = case.get("category")
        if not isinstance(category, str) or not CATEGORY_RE.fullmatch(category):
            add_error(errors, f"{path}.category", 'must match "主题-子主题" with 2-6 Chinese chars per part')

        answer = case.get("groundtruth_answer")
        if not isinstance(answer, str):
            add_error(errors, f"{path}.groundtruth_answer", "must be a string")
        else:
            if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", answer):
                add_error(errors, f"{path}.groundtruth_answer", "must contain a non-empty answer")
            for term in META_TERMS:
                if term in answer:
                    add_error(errors, f"{path}.groundtruth_answer", f"must not contain metadata term {term}")
            if LIST_ITEM_RE.search(answer):
                add_error(errors, f"{path}.groundtruth_answer", "must use prose, not a numbered or bullet list")

        intent = case.get("design_intent")
        if not isinstance(intent, str):
            add_error(errors, f"{path}.design_intent", "must be a string")
        elif not (intent.startswith("测试系统能否召回") or intent.startswith("测试系统能否围绕")):
            add_error(errors, f"{path}.design_intent", "must start with 测试系统能否召回 or 测试系统能否围绕")

    return errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Final dataset JSON to validate.")
    parser.add_argument("--knowledge-file", type=Path, nargs="+", help="Original knowledge JSON/JSONL files; verify that every id exists.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    known_ids = None
    if args.knowledge_file:
        from generate_qa_groundtruth import index_records, load_json_or_jsonl

        _, by_id = index_records([row for path in args.knowledge_file for row in load_json_or_jsonl(path)])
        known_ids = set(by_id)
    errors = validate_dataset(load_json(args.dataset), known_ids)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"validation ok: {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
