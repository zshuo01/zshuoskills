---
name: generate-knowledge-qa
description: 'Generate GraphRAG local-query evaluation datasets from atomic knowledge JSON/JSONL. Produces {dataset_name, source_file, design_principles, cases:[{query_id, category, question, groundtruth_knowledge_ids, groundtruth_answer, design_intent}]} from knowledge_id plus knowledge_text/content fields. Uses lexical local-topic clustering by default, exports LLM rewrite prompts, merges rewrites while preserving ids, and validates the final dataset.'
---

# Generate Knowledge QA

## Goal

Build a GraphRAG local-query evaluation dataset from atomic knowledge records. Each final case contains:

```json
{
  "query_id": "Q001",
  "category": "增值税-申报缴纳",
  "question": "在增值税申报缴纳事项中，应如何确定纳税主体、计税依据和申报缴纳？",
  "groundtruth_knowledge_ids": ["K001", "K002", "K003"],
  "groundtruth_answer": "纳税主体、计税依据和申报缴纳规则应以所列知识文本中的规定为准。申报缴纳事项需要同时覆盖纳税人、计税依据、期限或地点等要素。",
  "design_intent": "测试系统能否围绕增值税申报缴纳召回纳税主体、计税依据和申报缴纳相关规则。"
}
```

Important: the old two-field schema
`{"question": "...", "groundtruth_knowledge_ids": [...]}` is not a valid final
output for this skill. Use that shape only as an internal scratch note if
needed; never write it to the user-facing dataset.

## Core Rules

- First determine `groundtruth_knowledge_ids`; Python only selects ids/materials, while the LLM writes the natural `question`, `groundtruth_answer`, and final wording.
- Each case must contain 3-10 `groundtruth_knowledge_ids`; the maximum is always 10.
- Every listed id must serve the same local topic. No filler ids.
- Once ids enter the rewrite stage, they are fixed and must not be modified.
- If the fixed ids cannot support one natural Chinese question or a clear answer within 3 sentences, reject the candidate instead of adding, dropping, or replacing ids.
- `groundtruth_answer` is at most 3 sentences and must use only the supplied `supporting_knowledge` text. Do not add outside knowledge.
- The default candidate generator must not depend on external models, OpenAI APIs, embedding APIs, or network access. Lexical similarity is the default; embeddings may only be optional enhancements.
- The final JSON must pass `scripts/validate_qa_dataset.py`.
- The user-facing JSON file must contain only `dataset_name`, `source_file`, `design_principles`, and `cases`; each case must contain only `query_id`, `category`, `question`, `groundtruth_knowledge_ids`, `groundtruth_answer`, and `design_intent`.
- Before finishing, open or validate the final file and confirm it is a JSON object with `cases`, not a bare JSON array.

## Field Rules

- `category`: exactly `主题-子主题`; both parts should be concise Chinese labels, preferably 2-10 Chinese characters. Do not include `[草稿]`.
- `question`: one natural Chinese question with exactly one Chinese question mark `？`; no half-width `?`; no knowledge ids or metadata wording such as `原子知识`, `知识库`, `根据文本`, `根据材料`.
- `design_intent`: starts with `测试系统能否召回` or `测试系统能否围绕`, and names the local facets being tested.
- `groundtruth_answer`: at most 3 sentences, plain Chinese prose, no metadata wording, no bullet list.

## Workflow

1. Inspect input fields. Identify id fields (`knowledge_id`/`id`/`kid`), text fields (`knowledge_text`/`content`/`text`/`chunk`/`summary`), title fields, source buckets, and optional relation files.
2. Run `generate_qa_groundtruth.py` to generate candidate skeletons. By default it writes the full dataset object and the six allowed case fields, but `question`, `groundtruth_answer`, and `design_intent` are intentionally blank because Python must not template the natural wording:

```bash
python skills/generate-knowledge-qa/scripts/generate_qa_groundtruth.py 原子知识.json \
  -o qa_candidates.json --count 30 --min-ids 3 --target-ids 6 --max-ids 10
```

For LLM rewrite, explicitly include review fields so the prompt has supporting text:

```bash
python skills/generate-knowledge-qa/scripts/generate_qa_groundtruth.py 原子知识.json \
  -o qa_candidates_review.json --count 30 --include-review-fields
```

With a relation file:

```bash
python skills/generate-knowledge-qa/scripts/generate_qa_groundtruth.py 原子知识.json \
  --relations 原子知识文件关联.json -o qa_candidates.json --count 30
```

3. Export rewrite prompts:

```bash
python skills/generate-knowledge-qa/scripts/rewrite_and_merge.py qa_candidates_review.json \
  --export-prompts -o rewrite_prompts.jsonl
```

4. Use an LLM or external batch system to process `rewrite_prompts.jsonl`, producing `rewritten_results.jsonl`. Each line must preserve `query_id`, either as a top-level field or beside a nested `result` object.
5. Merge rewritten results into the final dataset:

```bash
python skills/generate-knowledge-qa/scripts/rewrite_and_merge.py qa_candidates_review.json \
  --merge rewritten_results.jsonl -o graphrag_local_eval_final.json
```

6. Validate:

```bash
python skills/generate-knowledge-qa/scripts/validate_qa_dataset.py graphrag_local_eval_final.json
```

7. Human spot-check: verify every id is needed, the question is natural, and the answer is fully grounded in the supporting texts.

## Candidate Generator Notes

`scripts/generate_qa_groundtruth.py` supports JSON arrays and JSONL, relation-assisted groups, same-source lexical groups, char 2-gram/3-gram similarity, facet overlap, topic consistency scoring, Jaccard overlap dedupe, and richer `supporting_knowledge`:

```json
{
  "knowledge_id": "K001",
  "knowledge_text": "较完整的文本，默认最多 3000 字",
  "preview_text": "前 180 字左右预览",
  "detected_facets": ["纳税主体", "申报缴纳"],
  "text_length": 1234
}
```

Review-only fields (`supporting_knowledge`, `quality_checks`) appear only when `--include-review-fields` is used. They must never appear in the final/user-facing dataset. The merged final dataset keeps only the six case fields.

## Common Failure To Avoid

If another installed copy of this skill says to output only `question` and
`groundtruth_knowledge_ids`, that copy is stale. Follow this file's required
GraphRAG dataset shape instead, then run `validate_qa_dataset.py`.

## Rewrite Output Format

The prompt itself asks the LLM to output one of the following objects.

Success:

```json
{
  "question": "...？",
  "category": "主题-子主题",
  "design_intent": "测试系统能否围绕...召回...",
  "groundtruth_answer": "..."
}
```

Reject:

```json
{
  "reject_reason": "这些 ids 无法共同支撑一个自然问题和三句话内答案。"
}
```

For `rewrite_and_merge.py`, keep the original `query_id` in the JSONL line, for example `{"query_id":"Q001","result":{...}}` or a flat object with `query_id` plus the rewritten fields. If a rewritten success object includes `groundtruth_knowledge_ids`, the merge script ignores it and preserves the candidate ids.
