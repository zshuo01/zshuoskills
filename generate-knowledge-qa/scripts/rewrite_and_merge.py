#!/usr/bin/env python3
"""Optional batch helper: export one prompt per pool, then merge chosen ids."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from validate_qa_dataset import CASE_KEYS, TOP_LEVEL_KEYS, nonempty_string, validate_dataset


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: invalid JSONL: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: result must be an object")
        rows.append(row)
    return rows


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def supporting_by_id(candidate: dict[str, Any]) -> dict[str, str]:
    supports = candidate.get("supporting_knowledge")
    if not isinstance(supports, list) or not supports:
        raise ValueError("candidate needs supporting_knowledge with complete source texts")
    result: dict[str, str] = {}
    for support in supports:
        if not isinstance(support, dict):
            raise ValueError("each supporting item must be an object")
        kid, text = support.get("knowledge_id"), support.get("knowledge_text")
        if not nonempty_string(kid) or not nonempty_string(text) or kid in result:
            raise ValueError("supporting ids must be unique non-empty strings with non-empty texts")
        # Also reject truncated materials from older candidate files.
        if support.get("is_truncated") or support.get("text_length", len(text)) != len(text):
            raise ValueError(f"{kid}: supporting text is truncated or its length is inconsistent")
        result[kid] = text
    if len(result) < 3:
        raise ValueError("candidate pool needs at least 3 knowledge records")
    return result


def candidate_cases(candidates: Any) -> list[dict[str, Any]]:
    if not isinstance(candidates, dict) or set(candidates) != set(TOP_LEVEL_KEYS):
        raise ValueError(f"candidate dataset fields must be exactly {TOP_LEVEL_KEYS}")
    cases = candidates.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("candidate dataset needs a non-empty cases list")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or not nonempty_string(case.get("query_id")):
            raise ValueError("each candidate needs a string query_id")
        if case["query_id"] in seen:
            raise ValueError(f"duplicate candidate query_id: {case['query_id']}")
        seen.add(case["query_id"])
        supporting_by_id(case)
    return cases


def export_prompts(candidates: dict[str, Any]) -> list[dict[str, Any]]:
    template = (Path(__file__).resolve().parents[1] / "references" / "rewrite_prompt.md").read_text(encoding="utf-8")
    template = template.split("```text", 1)[1].split("```", 1)[0].strip()
    return [
        {"query_id": case["query_id"], "prompt": template.replace("{candidate_json}", json.dumps({
            "query_id": case["query_id"], "category_hint": case.get("category", ""),
            "supporting_knowledge": case["supporting_knowledge"],
        }, ensure_ascii=False, indent=2))}
        for case in candidate_cases(candidates)
    ]


def normalize_rewrites(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    results = {}
    for row in rows:
        result = row
        for key in ("result", "rewrite", "rewritten", "output"):
            nested = row.get(key)
            if isinstance(nested, str):
                nested = json.loads(nested)
            if isinstance(nested, dict):
                result = dict(nested)
                if "query_id" in row:
                    if "query_id" in result and result["query_id"] != row["query_id"]:
                        raise ValueError("conflicting wrapper and nested query_id")
                    result["query_id"] = row["query_id"]
                break
        query_id = result.get("query_id")
        if not nonempty_string(query_id) or query_id in results:
            raise ValueError(f"missing or duplicate result query_id: {query_id}")
        results[query_id] = result
    return results


def merge_rewrites(candidates: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cases, results = candidate_cases(candidates), normalize_rewrites(rows)
    expected = {case["query_id"] for case in cases}
    if set(results) != expected:
        raise ValueError(f"missing query_ids={sorted(expected - set(results))}; unknown query_ids={sorted(set(results) - expected)}")
    final = {key: candidates[key] for key in TOP_LEVEL_KEYS}
    final["cases"], rejected = [], []
    for candidate in cases:
        result = results[candidate["query_id"]]
        reason = result.get("reject_reason")
        if reason is not None and not nonempty_string(reason):
            raise ValueError("reject_reason must be a non-empty string")
        if reason is None:
            # The model's chosen subset is the groundtruth, not the whole pool.
            case = {key: result.get(key) for key in CASE_KEYS}
            case["query_id"] = "Q001"
            errors = validate_dataset({**final, "cases": [case]}, set(supporting_by_id(candidate)))
            reason = "; ".join(errors) if errors else None
        if reason:
            rejected.append({"query_id": candidate["query_id"], "reject_reason": reason})
        else:
            case["query_id"] = f"Q{len(final['cases']) + 1:03d}"
            final["cases"].append(case)
    if final["cases"]:
        errors = validate_dataset(final)
        if errors:
            raise ValueError("; ".join(errors))
    return final, rejected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--export-prompts", action="store_true")
    mode.add_argument("--merge", type=Path, metavar="RESULTS_JSONL")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--rejected-output", type=Path, help="Optional rejection report; otherwise only print reasons.")
    args = parser.parse_args(argv)
    inputs = {path.resolve() for path in (args.candidates, args.merge) if path}
    if args.output.resolve() in inputs or (args.rejected_output and args.rejected_output.resolve() in inputs | {args.output.resolve()}):
        parser.error("output paths must differ from the inputs and each other")
    if args.merge and args.output.suffix.lower() != ".json":
        parser.error("final output must be a .json file")
    candidates = json.loads(args.candidates.read_text(encoding="utf-8-sig"))
    if args.export_prompts:
        prompts = export_prompts(candidates)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in prompts:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(prompts)} prompts to {args.output}")
        return 0
    final, rejected = merge_rewrites(candidates, read_jsonl(args.merge))
    for item in rejected:
        print(f"{item['query_id']}: {item['reject_reason']}", file=sys.stderr)
    if args.rejected_output:
        write_json({"rejected_cases": rejected}, args.rejected_output)
    if not final["cases"]:
        raise ValueError("no valid cases remain; final output was not written")
    write_json(final, args.output)
    print(f"wrote {len(final['cases'])} final cases to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from None
