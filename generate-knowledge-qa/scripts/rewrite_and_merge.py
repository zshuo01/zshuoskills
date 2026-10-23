#!/usr/bin/env python3
"""Export rewrite/review prompts, then merge reviewed cases into the fixed JSON schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from evidence_alignment import alignment_errors, review_errors, review_input_hash, supporting_by_id
from validate_qa_dataset import CASE_KEYS, TOP_LEVEL_KEYS, nonempty_string, validate_dataset

FINAL_CASE_KEYS = CASE_KEYS
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


def prompt_template(name: str = "rewrite_prompt.md") -> str:
    prompt_path = Path(__file__).resolve().parents[1] / "references" / name
    text = prompt_path.read_text(encoding="utf-8")
    if "```text" in text:
        text = text.split("```text", 1)[1].split("```", 1)[0].strip()
    return text


def render_prompt(case: dict[str, Any], template: str) -> str:
    supporting_by_id(case)
    supporting = json.dumps(case.get("supporting_knowledge", []), ensure_ascii=False, indent=2)
    local_topic = str(case.get("category", "局部主题"))
    return (template.replace("{query_id_json}", json.dumps(case["query_id"], ensure_ascii=False))
            .replace("{local_topic}", local_topic).replace("{supporting_knowledge}", supporting))


def candidate_cases(candidates: Any) -> list[dict[str, Any]]:
    if not isinstance(candidates, dict) or set(candidates) != set(TOP_LEVEL_KEYS):
        raise ValueError(f"candidate dataset fields must be exactly {TOP_LEVEL_KEYS}")
    cases = candidates.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("candidate dataset must contain a non-empty cases list")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not nonempty_string(case.get("query_id")):
            raise ValueError("each candidate must have a non-empty string query_id")
        query_id = case["query_id"]
        if query_id in seen:
            raise ValueError(f"duplicate candidate query_id: {query_id}")
        seen.add(query_id)
        try:
            supporting_by_id(case)
        except ValueError as exc:
            raise ValueError(f"{query_id}: {exc}") from exc
    return cases


def export_prompts(candidates: dict[str, Any]) -> list[dict[str, Any]]:
    template = prompt_template()
    rows: list[dict[str, Any]] = []
    for case in candidate_cases(candidates):
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
        elif isinstance(value, str):
            try:
                merged = json.loads(value)
            except json.JSONDecodeError:
                continue
            if not isinstance(merged, dict):
                continue
        else:
            continue
        for envelope_key in ("query_id", "input_hash"):
            if envelope_key in row:
                if envelope_key in merged and merged[envelope_key] != row[envelope_key]:
                    raise ValueError(f"conflicting wrapper and nested {envelope_key}")
                merged[envelope_key] = row[envelope_key]
        return merged
    return row


def normalize_rewrites(rows: list[dict[str, Any]], label: str = "rewritten_results") -> dict[str, dict[str, Any]]:
    rewrites: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows, start=1):
        rewrite = unwrap_rewrite(row)
        query_id = rewrite.get("query_id")
        if not nonempty_string(query_id):
            raise ValueError(f"{label} line {idx}: missing string query_id")
        if query_id in rewrites:
            raise ValueError(f"{label}: duplicate query_id {query_id}")
        rewrites[query_id] = rewrite
    return rewrites


def require_complete(results: dict[str, Any], expected: set[str], label: str) -> None:
    missing, unknown = sorted(expected - set(results)), sorted(set(results) - expected)
    if missing or unknown:
        raise ValueError(f"{label}: missing query_ids={missing}; unknown query_ids={unknown}")


def final_case_from_rewrite(
    new_query_id: str,
    candidate: dict[str, Any],
    rewrite: dict[str, Any],
) -> dict[str, Any]:
    missing = [key for key in SUCCESS_KEYS if not nonempty_string(rewrite.get(key))]
    if missing:
        raise ValueError(f"missing non-empty rewritten fields: {', '.join(sorted(missing))}")

    rewritten_ids = rewrite.get("groundtruth_knowledge_ids")
    original_ids = candidate.get("groundtruth_knowledge_ids", [])
    if rewritten_ids is not None and rewritten_ids != original_ids:
        raise ValueError("rewritten groundtruth_knowledge_ids differ from the fixed candidate ids")

    return {
        "query_id": new_query_id,
        "category": rewrite["category"],
        "question": rewrite["question"],
        "groundtruth_knowledge_ids": list(original_ids),
        "groundtruth_answer": rewrite["groundtruth_answer"],
        "design_intent": rewrite["design_intent"],
    }


def prepare_rewrites(
    candidates: dict[str, Any],
    rewritten_rows: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    cases = candidate_cases(candidates)
    rewrites = normalize_rewrites(rewritten_rows)
    require_complete(rewrites, {case["query_id"] for case in cases}, "rewritten_results")
    eligible = []
    rejected_cases: list[dict[str, Any]] = []
    for candidate in cases:
        rewrite = rewrites[candidate["query_id"]]
        reason = rewrite.get("reject_reason")
        if reason is not None and not nonempty_string(reason):
            raise ValueError(f"{candidate['query_id']}: reject_reason must be a non-empty string")
        if reason is None:
            try:
                case = final_case_from_rewrite("Q001", candidate, rewrite)
                errors = validate_dataset({**candidates, "cases": [case]})
                if not errors:
                    errors = alignment_errors(candidate, rewrite)
            except ValueError as exc:
                errors = [str(exc)]
            reason = "; ".join(errors) if errors else None
        if reason:
            rejected_cases.append({"query_id": candidate["query_id"],
                                   "groundtruth_knowledge_ids": candidate["groundtruth_knowledge_ids"],
                                   "reject_reason": reason})
        else:
            eligible.append((candidate, rewrite))
    return eligible, rejected_cases


def export_review_prompts(
    candidates: dict[str, Any], rewritten_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible, rejected = prepare_rewrites(candidates, rewritten_rows)
    template = prompt_template("alignment_review_prompt.md")
    rows = []
    for candidate, rewrite in eligible:
        input_hash = review_input_hash(candidate, rewrite)
        payload = {"query_id": candidate["query_id"], "input_hash": input_hash,
                   "groundtruth_knowledge_ids": candidate["groundtruth_knowledge_ids"],
                   "supporting_knowledge": candidate["supporting_knowledge"],
                   "proposed_case": final_case_from_rewrite(candidate["query_id"], candidate, rewrite),
                   "evidence_alignment": rewrite["evidence_alignment"]}
        prompt = template.replace("{review_payload}", json.dumps(payload, ensure_ascii=False, indent=2))
        rows.append({"query_id": candidate["query_id"], "input_hash": input_hash, "prompt": prompt})
    return rows, rejected


def merge_rewrites(
    candidates: dict[str, Any], rewritten_rows: list[dict[str, Any]],
    reviewed_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eligible, rejected_cases = prepare_rewrites(candidates, rewritten_rows)
    reviews = normalize_rewrites(reviewed_rows, "reviewed_results")
    require_complete(reviews, {candidate["query_id"] for candidate, _ in eligible}, "reviewed_results")
    final_cases: list[dict[str, Any]] = []
    for candidate, rewrite in eligible:
        review = reviews[candidate["query_id"]]
        if review.get("input_hash") != review_input_hash(candidate, rewrite):
            raise ValueError(f"{candidate['query_id']}: review input_hash mismatch; review the current materials and rewrite again")
        errors = review_errors(review)
        if errors:
            rejected_cases.append({"query_id": candidate["query_id"],
                                   "groundtruth_knowledge_ids": candidate["groundtruth_knowledge_ids"],
                                   "reject_reason": "; ".join(errors)})
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
    mode.add_argument("--export-review-prompts", type=Path, metavar="REWRITTEN_JSONL", help="Check evidence mappings and export a separate alignment-review batch.")
    mode.add_argument("--merge", type=Path, metavar="REWRITTEN_JSONL", help="Merge rewritten_results.jsonl.")
    parser.add_argument("--reviews", type=Path, help="Required with --merge: alignment review results JSONL.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output path.")
    parser.add_argument(
        "--rejected-output",
        type=Path,
        default=None,
        help="Rejected-case report path; defaults to <output>.rejected_cases.json when needed.",
    )
    args = parser.parse_args(argv)
    if bool(args.merge) != bool(args.reviews):
        parser.error("--merge requires --reviews; --reviews is only valid with --merge")
    if args.merge and args.output.suffix.lower() != ".json":
        parser.error("the final dataset output must be a .json file")
    input_paths = {path.resolve() for path in (args.candidates, args.merge, args.export_review_prompts, args.reviews) if path}
    rejected_path = args.rejected_output or args.output.with_suffix(args.output.suffix + ".rejected_cases.json")
    if args.output.resolve() in input_paths or rejected_path.resolve() in input_paths or rejected_path.resolve() == args.output.resolve():
        parser.error("output and rejection-report paths must differ from each other and all input files")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidates = read_json(args.candidates)

    if args.export_prompts:
        write_jsonl(export_prompts(candidates), args.output)
        print(f"wrote rewrite prompts to {args.output}")
        return 0

    rewritten_rows = read_jsonl(args.export_review_prompts or args.merge)
    if args.export_review_prompts:
        prompts, rejected_cases = export_review_prompts(candidates, rewritten_rows)
        final_dataset = None
    else:
        final_dataset, rejected_cases = merge_rewrites(candidates, rewritten_rows, read_jsonl(args.reviews))
    if rejected_cases:
        rejected_path = args.rejected_output or args.output.with_suffix(args.output.suffix + ".rejected_cases.json")
        write_json({"rejected_cases": rejected_cases}, rejected_path)
        print(f"wrote {len(rejected_cases)} rejected cases to {rejected_path}")
    if args.export_review_prompts:
        if not prompts:
            raise ValueError("no grounded rewrites remain; inspect rejected cases and regenerate")
        write_jsonl(prompts, args.output)
        print(f"wrote {len(prompts)} alignment-review prompts to {args.output}")
        return 0
    errors = validate_dataset(final_dataset)
    if errors:
        raise ValueError("final dataset was not written: " + "; ".join(errors))
    write_json(final_dataset, args.output)
    print(f"wrote {len(final_dataset['cases'])} final cases to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from None
