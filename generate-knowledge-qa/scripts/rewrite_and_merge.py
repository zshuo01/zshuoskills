#!/usr/bin/env python3
"""Export rewrite prompts or merge LLM rewrites into a final QA dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


FINAL_CASE_KEYS = [
    "query_id",
    "category",
    "question",
    "groundtruth_knowledge_ids",
    "groundtruth_answer",
    "design_intent",
]

SUCCESS_KEYS = {"question", "category", "design_intent", "groundtruth_answer"}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(value, dict):
                raise SystemExit(f"{path}:{line_no}: each line must be a JSON object")
            rows.append(value)
    return rows


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def prompt_template() -> str:
    prompt_path = Path(__file__).resolve().parents[1] / "references" / "rewrite_prompt.md"
    text = prompt_path.read_text(encoding="utf-8")
    if "```text" in text:
        text = text.split("```text", 1)[1].split("```", 1)[0].strip()
    return text


def render_prompt(case: dict[str, Any], template: str) -> str:
    supporting = json.dumps(case.get("supporting_knowledge", []), ensure_ascii=False, indent=2)
    local_topic = str(case.get("category", "局部主题"))
    return template.replace("{local_topic}", local_topic).replace("{supporting_knowledge}", supporting)


def export_prompts(candidates: dict[str, Any]) -> list[dict[str, Any]]:
    template = prompt_template()
    rows: list[dict[str, Any]] = []
    for case in candidates.get("cases", []):
        rows.append(
            {
                "query_id": case["query_id"],
                "local_topic": case.get("category", "局部主题"),
                "supporting_knowledge": case.get("supporting_knowledge", []),
                "groundtruth_knowledge_ids": case.get("groundtruth_knowledge_ids", []),
                "prompt": render_prompt(case, template),
            }
        )
    return rows


def unwrap_rewrite(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("rewrite", "rewritten", "result", "output"):
        value = row.get(key)
        if isinstance(value, dict):
            merged = dict(value)
            merged.setdefault("query_id", row.get("query_id"))
            return merged
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                parsed.setdefault("query_id", row.get("query_id"))
                return parsed
    return row


def normalize_rewrites(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rewrites: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows, start=1):
        rewrite = unwrap_rewrite(row)
        query_id = rewrite.get("query_id")
        if not query_id:
            raise SystemExit(f"rewritten_results line {idx}: missing query_id")
        rewrites[str(query_id)] = rewrite
    return rewrites


def final_case_from_rewrite(
    new_query_id: str,
    candidate: dict[str, Any],
    rewrite: dict[str, Any],
) -> dict[str, Any]:
    missing = [key for key in SUCCESS_KEYS if not rewrite.get(key)]
    if missing:
        raise SystemExit(f"{candidate['query_id']}: missing rewritten fields: {', '.join(sorted(missing))}")

    rewritten_ids = rewrite.get("groundtruth_knowledge_ids")
    original_ids = candidate.get("groundtruth_knowledge_ids", [])
    if rewritten_ids is not None and list(rewritten_ids) != list(original_ids):
        print(
            f"{candidate['query_id']}: rewritten groundtruth_knowledge_ids ignored; preserving candidate ids",
            file=sys.stderr,
        )

    return {
        "query_id": new_query_id,
        "category": rewrite["category"],
        "question": rewrite["question"],
        "groundtruth_knowledge_ids": original_ids,
        "groundtruth_answer": rewrite["groundtruth_answer"],
        "design_intent": rewrite["design_intent"],
    }


def merge_rewrites(
    candidates: dict[str, Any],
    rewritten_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rewrites = normalize_rewrites(rewritten_rows)
    final_cases: list[dict[str, Any]] = []
    rejected_cases: list[dict[str, Any]] = []

    for candidate in candidates.get("cases", []):
        original_query_id = candidate.get("query_id")
        rewrite = rewrites.get(str(original_query_id))
        if not rewrite:
            rejected_cases.append(
                {
                    "query_id": original_query_id,
                    "groundtruth_knowledge_ids": candidate.get("groundtruth_knowledge_ids", []),
                    "reject_reason": "missing rewritten result",
                }
            )
            continue
        if rewrite.get("reject_reason"):
            rejected_cases.append(
                {
                    "query_id": original_query_id,
                    "groundtruth_knowledge_ids": candidate.get("groundtruth_knowledge_ids", []),
                    "reject_reason": rewrite["reject_reason"],
                }
            )
            continue
        final_cases.append(
            final_case_from_rewrite(f"Q{len(final_cases) + 1:03d}", candidate, rewrite)
        )

    final_dataset = {
        "dataset_name": candidates.get("dataset_name", "graphrag_local_retrieval_eval"),
        "source_file": candidates.get("source_file", ""),
        "design_principles": candidates.get("design_principles", []),
        "cases": final_cases,
    }
    return final_dataset, rejected_cases


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path, help="qa_candidates.json from generate_qa_groundtruth.py.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--export-prompts", action="store_true", help="Write rewrite_prompts.jsonl.")
    mode.add_argument("--merge", type=Path, metavar="REWRITTEN_JSONL", help="Merge rewritten_results.jsonl.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output path.")
    parser.add_argument(
        "--rejected-output",
        type=Path,
        default=None,
        help="Rejected-case JSON path for --merge; defaults to <output>.rejected_cases.json when needed.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidates = read_json(args.candidates)

    if args.export_prompts:
        write_jsonl(export_prompts(candidates), args.output)
        print(f"wrote rewrite prompts to {args.output}")
        return 0

    rewritten_rows = read_jsonl(args.merge)
    final_dataset, rejected_cases = merge_rewrites(candidates, rewritten_rows)
    write_json(final_dataset, args.output)
    if rejected_cases:
        rejected_path = args.rejected_output or args.output.with_suffix(args.output.suffix + ".rejected_cases.json")
        write_json({"rejected_cases": rejected_cases}, rejected_path)
        print(f"wrote {len(rejected_cases)} rejected cases to {rejected_path}")
    print(f"wrote {len(final_dataset['cases'])} final cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
