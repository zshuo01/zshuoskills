#!/usr/bin/env python3
"""Generate GraphRAG local-query evaluation candidates from atomic knowledge.

The script is deliberately model-free by default. It fixes a 3-10 id local
topic cluster first, then emits draft wording and enough supporting text for a
later rewrite step. The rewrite step must preserve the ids or reject the case.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Sequence


ID_KEYS = ("knowledge_id", "id", "kid")
TITLE_KEYS = ("title", "heading", "name")
TEXT_KEYS = ("knowledge_text", "content", "text", "chunk", "summary")
GROUP_KEYS = ("source_file", "file_id", "document_id", "doc_id", "source", "title")
RELATION_PAIR_KEYS = (
    ("source", "target"),
    ("from", "to"),
    ("source_id", "target_id"),
    ("parent_id", "child_id"),
    ("start_id", "end_id"),
)

DEFAULT_MIN_IDS = 3
DEFAULT_TARGET_IDS = 6
DEFAULT_MAX_IDS = 10
DEFAULT_MAX_SUPPORTING_CHARS = 3000

TAX_TOPIC_WHITELIST = (
    "增值税",
    "消费税",
    "企业所得税",
    "个人所得税",
    "车船税",
    "印花税",
    "关税",
    "房产税",
    "契税",
    "资源税",
    "环境保护税",
    "城市维护建设税",
    "土地增值税",
    "城镇土地使用税",
    "耕地占用税",
    "车辆购置税",
    "船舶吨税",
)

TOPIC_ALIASES = {
    "环境保护税": "环保税",
    "城市维护建设税": "城建税",
    "城镇土地使用税": "土地使用税",
    "车辆购置税": "购置税",
}

STOPWORDS = {
    "纳税",
    "应纳税",
    "税额",
    "税率",
    "税款",
    "税法",
    "本法",
    "规定",
    "中华人民共和国",
    "依照",
    "有关",
    "应当",
    "原子知识",
    "知识库",
}

FACET_KEYWORDS: dict[str, tuple[str, ...]] = {
    "纳税主体": (
        "纳税人",
        "纳税义务人",
        "扣缴义务人",
        "居民企业",
        "非居民",
        "所有人",
        "管理人",
    ),
    "计税依据": (
        "计税依据",
        "应纳税所得额",
        "完税价格",
        "成交价格",
        "应税凭证",
        "销售额",
        "收入额",
    ),
    "税率税额": ("税率", "适用税率", "税额", "应纳税额", "税目"),
    "申报缴纳": ("申报", "缴纳", "纳税期限", "纳税地点", "申报期限", "缴款"),
    "减免优惠": ("免税", "减征", "减免", "优惠", "加计扣除", "抵免", "免征"),
    "扣缴源泉": ("扣缴", "源泉", "代扣代缴", "扣缴义务"),
    "登记权属": ("登记", "权属", "检验", "备案", "产权"),
}

SUBTHEME_COMBOS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("纳税主体", "申报缴纳"), "基础申报"),
    (("计税依据", "税率税额"), "计税规则"),
    (("申报缴纳", "计税依据"), "计税申报"),
    (("扣缴源泉", "申报缴纳"), "扣缴申报"),
    (("减免优惠", "申报缴纳"), "优惠申报"),
    (("减免优惠", "登记权属"), "减免征管"),
    (("登记权属", "申报缴纳"), "登记申报"),
    (("登记权属", "税率税额"), "登记计税"),
)

FACET_TO_SUBTHEME = {
    "纳税主体": "纳税主体",
    "计税依据": "计税依据",
    "税率税额": "税率税额",
    "申报缴纳": "申报缴纳",
    "减免优惠": "减免优惠",
    "扣缴源泉": "扣缴源泉",
    "登记权属": "登记权属",
}

DEFAULT_PRINCIPLES = [
    "每个问题围绕同一业务或法规主题组织，需要联合检索多个原子知识才能完整回答。",
    "groundtruth_knowledge_ids 仅保留回答该问题所必需的知识点。",
    "问题覆盖申报、征收、退税、抵扣、账户管理、法律责任等不同检索场景。",
    "问题采用自然中文表达，不在问题中直接暴露答案或 knowledge_id。",
]


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_json_or_jsonl(path: Path) -> list[Any]:
    text = read_text(path).strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        rows: list[Any] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
        return rows

    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("data", "items", "records", "knowledge", "cases"):
            if isinstance(value.get(key), list):
                return value[key]
        return [value]
    return []


def first_value(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def record_id(record: dict[str, Any]) -> str | None:
    value = first_value(record, ID_KEYS)
    return str(value) if value not in (None, "") else None


def normalize_space(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(value_to_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(value_to_text(item) for item in value.values())
    return str(value)


def normalized_text(record: dict[str, Any], max_chars: int = 3000) -> str:
    """Merge title-like and text-like fields with a bounded total length."""
    parts: list[str] = []
    for key in (*TITLE_KEYS, *TEXT_KEYS):
        if key not in record:
            continue
        value = normalize_space(value_to_text(record.get(key)))
        if value:
            parts.append(value[: max_chars])
    if not parts:
        for value in record.values():
            text = normalize_space(value_to_text(value))
            if text:
                parts.append(text[: max_chars])
    merged = normalize_space(" ".join(dict.fromkeys(parts)))
    return merged[:max_chars]


def full_normalized_text(record: dict[str, Any]) -> str:
    return normalized_text(record, max_chars=200_000)


def preview_text(text: str, limit: int = 180) -> str:
    return text if len(text) <= limit else f"{text[:limit].rstrip()}..."


def sort_key(kid: str) -> tuple[str, int, str]:
    match = re.search(r"^(.*?)(\d+)$", kid)
    if match:
        return (match.group(1), int(match.group(2)), kid)
    return (kid, 0, kid)


def cjk_runs(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]+", text)


def char_ngrams(text: str, sizes: Sequence[int] = (2, 3)) -> set[str]:
    features: set[str] = set()
    for run in cjk_runs(text):
        for size in sizes:
            if len(run) < size:
                continue
            features.update(run[i : i + size] for i in range(len(run) - size + 1))
    return features


def jaccard(left: set[str] | set[Any], right: set[str] | set[Any]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def detect_facets_from_text(text: str) -> list[str]:
    facets = []
    for label, keywords in FACET_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            facets.append(label)
    return facets


def detect_facets(records: list[dict[str, Any]]) -> list[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(detect_facets_from_text(normalized_text(record)))
    return [facet for facet, _ in counts.most_common()]


def canonical_topic(topic: str) -> str:
    topic = TOPIC_ALIASES.get(topic, topic)
    topic = "".join(re.findall(r"[\u4e00-\u9fff]+", topic))
    if 2 <= len(topic) <= 6:
        return topic
    if len(topic) > 6:
        return topic[:6]
    return "局部主题"


def topic_hits(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for topic in TAX_TOPIC_WHITELIST:
        hit_count = text.count(topic)
        if hit_count:
            counts[topic] += hit_count
    return counts


def cjk_terms(text: str) -> list[str]:
    terms: list[str] = []
    for run in cjk_runs(text):
        for size in (6, 5, 4, 3, 2):
            if len(run) >= size:
                terms.extend(run[i : i + size] for i in range(len(run) - size + 1))
    return terms


def record_topic(record: dict[str, Any]) -> str:
    title_text = " ".join(value_to_text(record.get(key)) for key in (*TITLE_KEYS, "source_file", "source"))
    hits = topic_hits(title_text)
    if hits:
        return canonical_topic(hits.most_common(1)[0][0])
    hits = topic_hits(normalized_text(record))
    if hits:
        return canonical_topic(hits.most_common(1)[0][0])
    return "局部主题"


def infer_main_topic(records: list[dict[str, Any]]) -> str:
    title_text = " ".join(
        value_to_text(record.get(key))
        for record in records
        for key in (*TITLE_KEYS, "source_file", "source")
    )
    title_hits = topic_hits(title_text)
    if title_hits:
        return canonical_topic(title_hits.most_common(1)[0][0])

    all_text = " ".join(normalized_text(record) for record in records)
    whitelist_hits = topic_hits(all_text)
    if whitelist_hits:
        return canonical_topic(whitelist_hits.most_common(1)[0][0])

    term_counts: Counter[str] = Counter()
    for record in records:
        for term in set(cjk_terms(normalized_text(record, max_chars=1200))):
            if 2 <= len(term) <= 6 and term not in STOPWORDS:
                if any(stop in term for stop in STOPWORDS):
                    continue
                term_counts[term] += 1
    if term_counts:
        return canonical_topic(term_counts.most_common(1)[0][0])
    return "局部主题"


def infer_subtheme(records: list[dict[str, Any]]) -> str:
    facet_counts: Counter[str] = Counter()
    for record in records:
        facet_counts.update(detect_facets_from_text(normalized_text(record)))
    if not facet_counts:
        return "综合规则"

    top_facets = [facet for facet, _ in facet_counts.most_common(3)]
    top_set = set(top_facets)
    for required, label in SUBTHEME_COMBOS:
        if set(required).issubset(top_set):
            return label
    return FACET_TO_SUBTHEME.get(top_facets[0], "综合规则")


def build_category(main_topic: str, subtheme: str) -> str:
    main = canonical_topic(main_topic)
    sub = canonical_topic(subtheme)
    return f"{main}-{sub}"


def build_question(main_topic: str, subtheme: str, facets: list[str]) -> str:
    """Do not synthesize natural questions in Python.

    The generator's job is to choose a coherent fixed id cluster. Natural
    wording belongs to the LLM rewrite step, which sees the fixed ids and their
    supporting text.
    """
    return ""


def build_design_intent(main_topic: str, subtheme: str, facets: list[str]) -> str:
    return ""


def extract_relation_edges(rows: list[Any], valid_ids: set[str]) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for left_key, right_key in RELATION_PAIR_KEYS:
            if left_key in row and right_key in row:
                add_edge(edges, valid_ids, row.get(left_key), row.get(right_key))
        ids = row.get("knowledge_ids") or row.get("ids")
        if isinstance(ids, list):
            ids = [str(item) for item in ids if str(item) in valid_ids]
            for idx, left in enumerate(ids):
                for right in ids[idx + 1 :]:
                    if left != right:
                        edges.add((left, right))
        single = row.get("knowledge_id")
        related = row.get("related_knowledge_id") or row.get("related_ids")
        if single is not None and related is not None:
            add_edge(edges, valid_ids, single, related)
    return edges


def add_edge(edges: set[tuple[str, str]], valid_ids: set[str], left: Any, right: Any) -> None:
    left_values = left if isinstance(left, list) else [left]
    right_values = right if isinstance(right, list) else [right]
    for left_item in left_values:
        for right_item in right_values:
            left_id, right_id = str(left_item), str(right_item)
            if left_id in valid_ids and right_id in valid_ids and left_id != right_id:
                edges.add((left_id, right_id))


def sized_windows(ids: list[str], min_ids: int, target_ids: int, max_ids: int) -> list[list[str]]:
    ids = sorted(dict.fromkeys(ids), key=sort_key)
    if len(ids) < min_ids:
        return []
    size = min(max(target_ids, min_ids), max_ids, len(ids))
    step = max(1, math.ceil(size / 2))
    groups = []
    for start in range(0, len(ids), step):
        group = ids[start : start + size]
        if len(group) >= min_ids:
            groups.append(group)
    return groups


def relation_groups(edges: set[tuple[str, str]], min_ids: int, target_ids: int, max_ids: int) -> list[list[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        graph[left].add(right)
        graph[right].add(left)

    groups: list[list[str]] = []
    visited: set[str] = set()
    for start in sorted(graph, key=sort_key):
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        component: list[str] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for nxt in sorted(graph[node], key=sort_key):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        groups.extend(sized_windows(component, min_ids, target_ids, max_ids))
    return groups


def bucket_key(record: dict[str, Any]) -> str:
    value = first_value(record, GROUP_KEYS)
    return normalize_space(value_to_text(value)) or "all"


def bucket_groups(
    records: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    min_ids: int,
    target_ids: int,
    max_ids: int,
) -> list[list[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for record in records:
        kid = record_id(record)
        if kid:
            buckets[bucket_key(record)].append(kid)

    groups: list[list[str]] = []
    for ids in buckets.values():
        sorted_ids = sorted(dict.fromkeys(ids), key=sort_key)
        groups.extend(sized_windows(sorted_ids, min_ids, target_ids, max_ids))
        groups.extend(lexical_neighbor_groups(sorted_ids, by_id, min_ids, target_ids, max_ids))
    return groups


def lexical_neighbor_groups(
    ids: list[str],
    by_id: dict[str, dict[str, Any]],
    min_ids: int,
    target_ids: int,
    max_ids: int,
) -> list[list[str]]:
    if len(ids) < min_ids:
        return []

    texts = {kid: normalized_text(by_id[kid]) for kid in ids}
    grams = {kid: char_ngrams(texts[kid]) for kid in ids}
    facets = {kid: set(detect_facets_from_text(texts[kid])) for kid in ids}
    topics = {kid: record_topic(by_id[kid]) for kid in ids}

    groups: list[list[str]] = []
    for seed in ids:
        scored: list[tuple[float, str]] = []
        for other in ids:
            if other == seed:
                continue
            lexical = jaccard(grams[seed], grams[other])
            facet = jaccard(facets[seed], facets[other])
            topic = 1.0 if topics[seed] == topics[other] and topics[seed] != "局部主题" else 0.0
            pair_score = 0.6 * lexical + 0.25 * facet + 0.15 * topic
            if lexical >= 0.03 or (facet > 0 and topic > 0) or pair_score >= 0.12:
                scored.append((pair_score, other))
        selected = [seed] + [kid for _, kid in sorted(scored, reverse=True)[: max_ids - 1]]
        selected = sorted(dict.fromkeys(selected[: max(target_ids, min_ids)]), key=sort_key)
        if len(selected) >= min_ids:
            groups.append(selected)
    return groups


def pairwise_average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def score_cluster(ids: list[str], by_id: dict[str, dict[str, Any]]) -> dict[str, float]:
    texts = [normalized_text(by_id[kid]) for kid in ids]
    gram_sets = [char_ngrams(text) for text in texts]
    facet_sets = [set(detect_facets_from_text(text)) for text in texts]
    topics = [record_topic(by_id[kid]) for kid in ids]

    lexical_pairs: list[float] = []
    facet_pairs: list[float] = []
    for idx in range(len(ids)):
        for jdx in range(idx + 1, len(ids)):
            lexical_pairs.append(jaccard(gram_sets[idx], gram_sets[jdx]))
            facet_pairs.append(jaccard(facet_sets[idx], facet_sets[jdx]))

    lexical_score = pairwise_average(lexical_pairs)
    facet_score = pairwise_average(facet_pairs)
    topic_counts = Counter(topic for topic in topics if topic != "局部主题")
    topic_score = topic_counts.most_common(1)[0][1] / len(ids) if topic_counts else 0.0
    final_score = 0.55 * lexical_score + 0.25 * facet_score + 0.20 * topic_score
    return {
        "lexical_similarity_score": round(lexical_score, 4),
        "facet_overlap_score": round(facet_score, 4),
        "topic_consistency_score": round(topic_score, 4),
        "final_coherence_score": round(final_score, 4),
    }


def dedupe_clusters(
    clusters: list[dict[str, Any]],
    overlap_threshold: float = 0.75,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for cluster in sorted(
        clusters,
        key=lambda item: (-item["scores"]["final_coherence_score"], tuple(item["ids"])),
    ):
        cluster_ids = set(cluster["ids"])
        if any(jaccard(cluster_ids, set(existing["ids"])) >= overlap_threshold for existing in kept):
            continue
        kept.append(cluster)
    return kept


def make_supporting_knowledge(
    kid: str,
    record: dict[str, Any],
    max_supporting_chars: int,
) -> dict[str, Any]:
    full_text = full_normalized_text(record)
    support_text = full_text[:max_supporting_chars]
    return {
        "knowledge_id": kid,
        "knowledge_text": support_text,
        "preview_text": preview_text(full_text),
        "detected_facets": detect_facets_from_text(full_text),
        "text_length": len(full_text),
    }


def make_case(
    query_id: str,
    ids: list[str],
    by_id: dict[str, dict[str, Any]],
    final_only: bool,
    max_supporting_chars: int = DEFAULT_MAX_SUPPORTING_CHARS,
    scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    ids = [kid for kid in ids if kid in by_id]
    records = [by_id[kid] for kid in ids]
    main_topic = infer_main_topic(records)
    subtheme = infer_subtheme(records)
    facets = detect_facets(records)

    case: dict[str, Any] = {
        "query_id": query_id,
        "category": build_category(main_topic, subtheme),
        "question": build_question(main_topic, subtheme, facets),
        "groundtruth_knowledge_ids": ids,
        "groundtruth_answer": "",
        "design_intent": build_design_intent(main_topic, subtheme, facets),
    }
    if not final_only:
        case["supporting_knowledge"] = [
            make_supporting_knowledge(kid, by_id[kid], max_supporting_chars)
            for kid in ids
        ]
        case["quality_checks"] = {
            "groundtruth_count": len(ids),
            "detected_facets": facets,
            **(scores or score_cluster(ids, by_id)),
        }
    return case


def write_dataset(dataset: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def validate_id_args(min_ids: int, target_ids: int, max_ids: int) -> None:
    if not (3 <= min_ids <= target_ids <= max_ids <= 10):
        raise SystemExit("--min-ids/--target-ids/--max-ids must satisfy 3 <= min <= target <= max <= 10")


def build_candidates(
    records: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    valid_ids = set(by_id)
    raw_groups: list[tuple[str, list[str]]] = []

    if args.relations and args.relations.exists():
        relation_rows = load_json_or_jsonl(args.relations)
        edges = extract_relation_edges(relation_rows, valid_ids)
        raw_groups.extend(
            ("relation", group)
            for group in relation_groups(edges, args.min_ids, args.target_ids, args.max_ids)
        )

    raw_groups.extend(
        ("bucket", group)
        for group in bucket_groups(records, by_id, args.min_ids, args.target_ids, args.max_ids)
    )

    scored_clusters: list[dict[str, Any]] = []
    seen_exact: set[tuple[str, ...]] = set()
    for origin, group in raw_groups:
        ids = sorted([kid for kid in dict.fromkeys(group) if kid in valid_ids], key=sort_key)
        if not (args.min_ids <= len(ids) <= args.max_ids):
            continue
        key = tuple(ids)
        if key in seen_exact:
            continue
        seen_exact.add(key)
        scores = score_cluster(ids, by_id)
        if scores["final_coherence_score"] < args.min_coherence:
            continue
        scored_clusters.append({"ids": ids, "origin": origin, "scores": scores})

    deduped = dedupe_clusters(scored_clusters, args.overlap_threshold)
    cases: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    for cluster in deduped:
        records_in_group = [by_id[kid] for kid in cluster["ids"]]
        category = build_category(infer_main_topic(records_in_group), infer_subtheme(records_in_group))
        if args.max_per_category and category_counts[category] >= args.max_per_category:
            continue
        query_id = f"Q{len(cases) + 1:03d}"
        case = make_case(
            query_id,
            cluster["ids"],
            by_id,
            final_only=args.final_only,
            max_supporting_chars=args.max_supporting_chars,
            scores=cluster["scores"],
        )
        if not args.final_only:
            case["quality_checks"]["cluster_origin"] = cluster["origin"]
        cases.append(case)
        category_counts[category] += 1
        if len(cases) >= args.count:
            break
    return cases


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("knowledge_file", type=Path, help="Atomic knowledge JSON or JSONL file.")
    parser.add_argument("--relations", type=Path, help="Optional relation JSON/JSONL file linking knowledge ids.")
    parser.add_argument("-o", "--output", type=Path, default=Path("qa_candidates.json"), help="Output candidate JSON.")
    parser.add_argument("--count", type=int, default=30, help="Maximum number of cases to emit.")
    parser.add_argument("--min-ids", type=int, default=DEFAULT_MIN_IDS, help="Minimum ids per case; must be >= 3.")
    parser.add_argument("--target-ids", type=int, default=DEFAULT_TARGET_IDS, help="Preferred ids per case.")
    parser.add_argument("--max-ids", type=int, default=DEFAULT_MAX_IDS, help="Maximum ids per case; must be <= 10.")
    parser.add_argument("--dataset-name", default=None, help="dataset_name field; defaults to graphrag_local_retrieval_eval_<N>.")
    parser.add_argument("--source-file", default=None, help="source_file field; defaults to the input file name.")
    parser.add_argument(
        "--include-review-fields",
        action="store_true",
        help="Include supporting_knowledge and quality_checks for rewrite/review; final JSON should omit them.",
    )
    parser.add_argument("--final-only", action="store_true", help="Deprecated compatibility flag; this is now the default.")
    parser.add_argument("--max-supporting-chars", type=int, default=DEFAULT_MAX_SUPPORTING_CHARS, help="Max chars per supporting knowledge_text.")
    parser.add_argument("--min-coherence", type=float, default=0.12, help="Minimum final_coherence_score for output clusters.")
    parser.add_argument("--overlap-threshold", type=float, default=0.75, help="Jaccard id-overlap threshold for candidate dedupe.")
    parser.add_argument("--max-per-category", type=int, default=2, help="At most N cases per category; 0 disables the cap.")
    parser.add_argument("--allow-nonexclusive-topic", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.final_only = (not args.include_review_fields) or args.final_only
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_id_args(args.min_ids, args.target_ids, args.max_ids)
    if args.count < 1:
        raise SystemExit("--count must be positive")
    if args.max_supporting_chars < 500:
        raise SystemExit("--max-supporting-chars should be at least 500")

    raw_records = load_json_or_jsonl(args.knowledge_file)
    records = [row for row in raw_records if isinstance(row, dict) and record_id(row)]
    by_id = {record_id(row): row for row in records if record_id(row)}
    if len(by_id) < args.min_ids:
        raise SystemExit(f"Need at least {args.min_ids} records with knowledge_id/id/kid")

    cases = build_candidates(records, by_id, args)
    dataset = {
        "dataset_name": args.dataset_name or f"graphrag_local_retrieval_eval_{len(cases)}",
        "source_file": args.source_file or args.knowledge_file.name,
        "design_principles": DEFAULT_PRINCIPLES,
        "cases": cases,
    }
    write_dataset(dataset, args.output)
    print(f"wrote {len(cases)} cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
