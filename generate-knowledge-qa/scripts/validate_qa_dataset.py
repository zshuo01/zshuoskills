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
FORBIDDEN_ANYWHERE = ("supporting_knowledge", "quality_checks", "[草稿]")
CATEGORY_RE = re.compile(r"^[\u4e00-\u9fff]{2,6}-[\u4e00-\u9fff]{2,6}$")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def count_sentences(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    endings = re.findall(r"[。！？!?]", text)
    return len(endings) if endings else 1


def add_error(errors: list[str], path: str, message: str) -> None:
    errors.append(f"{path}: {message}")


def contains_forbidden(value: Any, term: str) -> bool:
    if isinstance(value, str):
        return term in value
    if isinstance(value, list):
        return any(contains_forbidden(item, term) for item in value)
    if isinstance(value, dict):
        return any(term in str(key) or contains_forbidden(item, term) for key, item in value.items())
    return False


def validate_dataset(data: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["$: dataset must be a JSON object"]

    for key in TOP_LEVEL_KEYS:
        if key not in data:
            add_error(errors, key, "missing required top-level field")

    for term in FORBIDDEN_ANYWHERE:
        if contains_forbidden(data, term):
            add_error(errors, "$", f"must not contain {term}")

    cases = data.get("cases")
    if not isinstance(cases, list):
        add_error(errors, "cases", "must be a list")
        return errors

    for index, case in enumerate(cases):
        path = f"cases[{index}]"
        if not isinstance(case, dict):
            add_error(errors, path, "must be an object")
            continue
        if list(case.keys()) != CASE_KEYS:
            add_error(errors, path, f"fields must be exactly {CASE_KEYS}")

        expected_query_id = f"Q{index + 1:03d}"
        if case.get("query_id") != expected_query_id:
            add_error(errors, f"{path}.query_id", f"must be {expected_query_id}")

        ids = case.get("groundtruth_knowledge_ids")
        if not isinstance(ids, list):
            add_error(errors, f"{path}.groundtruth_knowledge_ids", "must be a list")
        else:
            if not 3 <= len(ids) <= 10:
                add_error(errors, f"{path}.groundtruth_knowledge_ids", "must contain 3-10 ids")
            if len(ids) != len(set(ids)):
                add_error(errors, f"{path}.groundtruth_knowledge_ids", "must not contain duplicates")

        question = case.get("question")
        if not isinstance(question, str):
            add_error(errors, f"{path}.question", "must be a string")
        else:
            if question.count("？") != 1:
                add_error(errors, f"{path}.question", "must contain exactly one Chinese question mark")
            if "?" in question:
                add_error(errors, f"{path}.question", "must not contain half-width question mark")
            for term in META_TERMS:
                if term in question:
                    add_error(errors, f"{path}.question", f"must not contain metadata term {term}")

        category = case.get("category")
        if not isinstance(category, str) or not CATEGORY_RE.match(category):
            add_error(errors, f"{path}.category", 'must match "主题-子主题" with 2-6 Chinese chars per part')

        answer = case.get("groundtruth_answer")
        if not isinstance(answer, str):
            add_error(errors, f"{path}.groundtruth_answer", "must be a string")
        else:
            if count_sentences(answer) > 3:
                add_error(errors, f"{path}.groundtruth_answer", "must be at most 3 sentences")
            for term in META_TERMS:
                if term in answer:
                    add_error(errors, f"{path}.groundtruth_answer", f"must not contain metadata term {term}")

        intent = case.get("design_intent")
        if not isinstance(intent, str):
            add_error(errors, f"{path}.design_intent", "must be a string")
        elif not (intent.startswith("测试系统能否召回") or intent.startswith("测试系统能否围绕")):
            add_error(errors, f"{path}.design_intent", "must start with 测试系统能否召回 or 测试系统能否围绕")

    return errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Final dataset JSON to validate.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    errors = validate_dataset(load_json(args.dataset))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"validation ok: {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
