---
name: generate-knowledge-qa
description: 'Generate Chinese financial GraphRAG local-query evaluation datasets from atomic knowledge JSON/JSONL. Select fixed knowledge ids, write questions and grounded answers, check each id’s necessary contribution, and review alignment before producing the established JSON dataset schema.'
---

# Generate Knowledge QA

Generate questions that match both their `groundtruth_knowledge_ids` and
`groundtruth_answer`. Same-topic knowledge alone is insufficient: every listed
id must contribute information needed to answer the actual question.

## Final output contract

The deliverable is a **JSON file containing one object**, never JSONL, a bare
array, or a prose-only response. Keep the existing schema exactly:

```json
{
  "dataset_name": "graphrag_local_retrieval_eval_1",
  "source_file": "knowledge.json",
  "design_principles": ["固定知识集合应充分支撑问题和答案，每个知识点都有必要贡献。"],
  "cases": [
    {
      "query_id": "Q001",
      "category": "理财业务-申请办理",
      "question": "申购星河产品需要具备什么资格、提交哪些材料，并在什么时间内申请？",
      "groundtruth_knowledge_ids": ["K001", "K002", "K003"],
      "groundtruth_answer": "申购人须为已完成风险测评的个人客户。申请人须提交有效身份证明，且须在交易日15:00前提交申请。",
      "design_intent": "测试系统能否围绕产品申购召回客户资格、申请材料和办理时间。"
    }
  ]
}
```

The product and rules above are fictional examples. Actual answers must come
from the supplied knowledge. The only top-level fields are `dataset_name`,
`source_file`, `design_principles`, `cases`; the only case fields are the six
shown above, written in that order. Never add evidence, review, hash, score, or
graph fields to the final file. Intermediate JSON/JSONL may contain review data.

## Invariants

- Python selects ids/materials; the LLM writes natural questions and answers.
- Each case has 3–10 unique ids. Fix ids before rewrite; do not add, drop,
  replace, or reorder them. Reject unsuitable combinations rather than forcing
  a question or silently changing the ids.
- The fixed set must suffice to answer every part of the question. Each id
  must contribute a necessary fact; duplicates and related background are not
  valid filler. Test necessity by considering removal of each id.
- Answers directly and completely answer the question in at most 3 Chinese
  prose sentences. Preserve conditions, exceptions, negation, dates, scope,
  numbers, and units. Avoid generic answers such as “以相关规定为准”.
- Use only supplied knowledge and explicit question premises. Any calculation
  must identify its inputs and derivation and be independently rechecked.
- `category` is `主题-子主题`, each part 2–6 Chinese characters. `question`
  contains exactly one `？`, no `?`, knowledge ids, or metadata wording such as
  `原子知识`, `知识库`, `根据文本`, `根据材料`.
- `design_intent` starts with `测试系统能否召回` or `测试系统能否围绕` and identifies
  the information being tested. The answer contains no metadata or bullet list.
- Candidate selection remains offline and model-free by default. A relation
  file is an optional grouping input; there is no graph-structure requirement.

## Workflow

Commands below use `<skill-dir>` to mean the folder containing this SKILL.md;
resolve it to the actual installed or repository path before running.

1. Inspect input. Supported ids: `knowledge_id`/`id`/`kid`; text:
   `knowledge_text`/`content`/`text`/`chunk`/`summary`. Duplicate ids and missing
   text are errors. Identify whether time, jurisdiction, or version conditions
   are present; do not discard them when writing questions.
2. Generate review candidates with full supporting texts:

   ```bash
   python <skill-dir>/scripts/generate_qa_groundtruth.py knowledge.json -o qa_candidates_review.json --count 30 --include-review-fields
   ```

   Optional `--relations relations.json` assists grouping as before. Defaults
   remain min/target/max ids 3/6/10 and at most two candidates per provisional
   category. `--count` is an upper bound, not a guaranteed final count.
   `--max-supporting-chars 0` (default) keeps full text; truncated review
   materials cannot enter rewrite. Question, answer, and intent are blank
   candidate fields, not a finished dataset. Compact candidates without
   `--include-review-fields` cannot be used for rewrite.
3. Export prompts, then process every row using the LLM. Follow
   [rewrite_prompt.md](references/rewrite_prompt.md): success includes an
   intermediate `evidence_alignment` linking each fixed id to an exact source
   quote, a question span, an answer span, and its necessary contribution.
   Return a concrete `reject_reason` when the set cannot support a valid case.

   ```bash
   python <skill-dir>/scripts/rewrite_and_merge.py qa_candidates_review.json --export-prompts -o rewrite_prompts.jsonl
   ```

   Save one result per candidate in `rewritten_results.jsonl`, preserving
   `query_id`. No missing, duplicate, or unknown ids are accepted.
4. Export a separate review batch:

   ```bash
   python <skill-dir>/scripts/rewrite_and_merge.py qa_candidates_review.json --export-review-prompts rewritten_results.jsonl -o alignment_review_prompts.jsonl
   ```

   This checks actual quotes, coverage of all ids and the full answer, fixed-id
   integrity, and uncited direct numbers. A `derivation` explains a calculation;
   it is not proof that the calculation is correct.
5. Process the review prompts as a fresh review pass using
   [alignment_review_prompt.md](references/alignment_review_prompt.md). Re-read
   the original knowledge rather than accepting the writer’s explanation.
   Check answerability, answer support, necessity of every id, scope, and
   numerical correctness. Copy `query_id` and `input_hash` into
   `reviewed_results.jsonl`. Do not fabricate review approvals or hashes;
   changing any material, wording, or evidence mapping requires a new review.
6. Merge only reviewed, accepted cases into the final JSON:

   ```bash
   python <skill-dir>/scripts/rewrite_and_merge.py qa_candidates_review.json --merge rewritten_results.jsonl --reviews reviewed_results.jsonl -o graphrag_local_eval_final.json
   python <skill-dir>/scripts/validate_qa_dataset.py graphrag_local_eval_final.json --knowledge-file knowledge.json
   ```

   Merge automatically validates the final schema before writing. Rejected
   cases go to a separate report; an empty result is an error. Rejection may
   reduce the count: inspect reasons and regenerate when more cases are needed,
   without relaxing evidence requirements or padding ids. Final query ids are
   numbered sequentially as before.
7. Spot-check complete cases against original knowledge, especially numbers,
   exceptions, and whether each id is necessary. Open the final file to confirm
   the four top-level and six case fields, then deliver that JSON file.

Structural validation and exact-quote checks cannot prove semantic entailment.
The separate review and spot-check are part of producing trustworthy groundtruth.
