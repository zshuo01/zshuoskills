#!/usr/bin/env python3
"""Check intermediate evidence mappings; no review fields enter the final JSON.

These checks establish traceability, not semantic entailment. A separate review
must still judge answerability, faithful interpretation, and id necessity.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from decimal import Decimal
from typing import Any

from validate_qa_dataset import nonempty_string


REVIEW_CHECKS = (
    "question_answerable",
    "answer_supported",
    "all_ids_necessary",
    "scope_consistent",
    "numbers_correct",
)
ALIGNMENT_KEYS = {"knowledge_id", "question_span", "answer_span", "evidence_quote", "necessity"}
NUMBER_RE = re.compile(r"(?<!\d)[+-]?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|‰)?")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def supporting_by_id(candidate: dict[str, Any]) -> dict[str, str]:
    ids = candidate.get("groundtruth_knowledge_ids")
    if not isinstance(ids, list) or not 3 <= len(ids) <= 10 or not all(nonempty_string(kid) for kid in ids):
        raise ValueError("candidate must have 3-10 non-empty string ids")
    if any(kid != kid.strip() for kid in ids) or len(set(ids)) != len(ids):
        raise ValueError("candidate ids must be unique and have no outer whitespace")
    supports = candidate.get("supporting_knowledge")
    if not isinstance(supports, list) or not supports:
        raise ValueError("missing supporting_knowledge; regenerate with --include-review-fields")
    by_id: dict[str, str] = {}
    for support in supports:
        if not isinstance(support, dict):
            raise ValueError("each supporting_knowledge item must be an object")
        kid = support.get("knowledge_id")
        text = support.get("knowledge_text")
        if not nonempty_string(kid) or not nonempty_string(text):
            raise ValueError("each supporting item needs a knowledge_id and non-empty knowledge_text")
        if kid in by_id:
            raise ValueError(f"duplicate supporting id: {kid}")
        length = support.get("text_length", len(text))
        if not isinstance(length, int) or isinstance(length, bool) or length < len(text):
            raise ValueError(f"{kid}: invalid text_length")
        if support.get("is_truncated") or length > len(text):
            raise ValueError(f"{kid}: supporting text is truncated; regenerate with --max-supporting-chars 0")
        by_id[kid] = text
    if set(by_id) != set(ids):
        raise ValueError("supporting_knowledge ids must exactly match groundtruth_knowledge_ids")
    return by_id


def number_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in NUMBER_RE.finditer(unicodedata.normalize("NFKC", text)):
        token = re.sub(r"[\s,]", "", match.group())
        unit = token[-1] if token.endswith(("%", "‰")) else ""
        value = token[:-1] if unit else token
        canonical = format(Decimal(value), "f")
        if "." in canonical:
            canonical = canonical.rstrip("0").rstrip(".")
        tokens.add(f"{'0' if canonical == '-0' else canonical}{unit}")
    return tokens


def alignment_errors(candidate: dict[str, Any], rewrite: dict[str, Any]) -> list[str]:
    supports = supporting_by_id(candidate)
    rows = rewrite.get("evidence_alignment")
    if not isinstance(rows, list) or not rows:
        return ["missing evidence_alignment; map every fixed id to the question, answer, and an exact quote"]
    question = normalize_space(rewrite["question"])
    answer = normalize_space(rewrite["groundtruth_answer"])
    errors: list[str] = []
    used_ids: set[str] = set()
    covered: set[int] = set()
    claim_quotes: dict[str, list[str]] = {}
    derived_claims: set[str] = set()
    for index, row in enumerate(rows):
        path = f"evidence_alignment[{index}]"
        if not isinstance(row, dict) or not ALIGNMENT_KEYS <= set(row):
            errors.append(f"{path}: requires {sorted(ALIGNMENT_KEYS)}")
            continue
        if not all(nonempty_string(row[key]) for key in ALIGNMENT_KEYS):
            errors.append(f"{path}: all mapping fields must be non-empty strings")
            continue
        kid = row["knowledge_id"]
        if kid not in supports:
            errors.append(f"{path}: unknown or non-fixed knowledge_id {kid}")
            continue
        used_ids.add(kid)
        quote = normalize_space(row["evidence_quote"])
        if not any(char.isalnum() for char in quote) or quote not in normalize_space(supports[kid]):
            errors.append(f"{path}: evidence_quote is not an exact source passage for {kid}")
        question_span = normalize_space(row["question_span"])
        if not any(char.isalnum() for char in question_span) or question_span not in question:
            errors.append(f"{path}: question_span must quote a meaningful part of the question")
        span = normalize_space(row["answer_span"])
        if not any(char.isalnum() for char in span) or span not in answer:
            errors.append(f"{path}: answer_span must quote a meaningful part of the answer")
            continue
        for match in re.finditer(re.escape(span), answer):
            covered.update(range(match.start(), match.end()))
        claim_quotes.setdefault(span, []).append(quote)
        if "derivation" in row:
            if not nonempty_string(row["derivation"]):
                errors.append(f"{path}: derivation must explain a calculation using the cited facts and question premises")
            else:
                derived_claims.add(span)

    unused = sorted(set(supports) - used_ids)
    if unused:
        errors.append(f"fixed ids have no question/answer contribution: {unused}")
    uncovered = "".join(char for index, char in enumerate(answer) if index not in covered and (char.isalnum() or char in "%％‰≤≥<>"))
    if uncovered:
        errors.append(f"answer contains unmapped content: {uncovered}")
    for span, quotes in claim_quotes.items():
        unsupported = number_tokens(span) - number_tokens(" ".join(quotes))
        if unsupported and span not in derived_claims:
            errors.append(f"answer_span has numbers absent from its cited passages: {sorted(unsupported)}; a genuine calculation needs an explicit derivation and review")
    return errors


def review_input_hash(candidate: dict[str, Any], rewrite: dict[str, Any]) -> str:
    payload = {
        "query_id": candidate["query_id"],
        "groundtruth_knowledge_ids": candidate["groundtruth_knowledge_ids"],
        "supporting_knowledge": candidate["supporting_knowledge"],
        "rewrite": {key: rewrite.get(key) for key in ("question", "category", "groundtruth_answer", "design_intent", "evidence_alignment")},
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def review_errors(review: dict[str, Any]) -> list[str]:
    errors = []
    if review.get("verdict") not in ("accept", "reject"):
        errors.append("review verdict must be accept or reject")
    if not nonempty_string(review.get("reason")):
        errors.append("review must give a non-empty reason")
    checks = review.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(REVIEW_CHECKS):
        errors.append(f"review checks must be exactly {list(REVIEW_CHECKS)}")
    else:
        for key in REVIEW_CHECKS:
            if not isinstance(checks[key], bool):
                errors.append(f"review check {key} must be a boolean")
            elif checks[key] is not True:
                errors.append(f"review failed: {key}")
    if review.get("verdict") == "reject":
        errors.append(f"review rejected: {review.get('reason', '')}")
    return errors
