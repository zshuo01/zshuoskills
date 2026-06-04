# Generate Knowledge QA

Generate GraphRAG local-query evaluation datasets from atomic knowledge JSON/JSONL.

## Required Output Shape

The user-facing JSON contains only:

- Top level: `dataset_name`, `source_file`, `design_principles`, `cases`
- Each case: `query_id`, `category`, `question`, `groundtruth_knowledge_ids`, `groundtruth_answer`, `design_intent`

No `supporting_knowledge`, `quality_checks`, prompts, scores, or other review fields may appear in the final/user-facing JSON.

Do not use the legacy two-field final shape
`{"question": "...", "groundtruth_knowledge_ids": [...]}`. That shape is
insufficient for this GraphRAG local-query evaluation skill.

## Pipeline

1. Inspect input fields.
2. Run `generate_qa_groundtruth.py` to generate candidate skeletons in the full dataset object shape. Python fixes ids/materials only; it does not template natural questions.
3. For rewrite work only, rerun with `--include-review-fields` and export prompts.
4. Use an LLM to process prompts and produce `rewritten_results.jsonl`.
5. Run `rewrite_and_merge.py --merge` to build the final dataset.
6. Run `validate_qa_dataset.py` to validate the final JSON.
7. Human spot-check the final cases.

## Commands

Compact candidate JSON with no review fields. It keeps the six allowed case fields, but natural wording fields stay blank until LLM rewrite:

```bash
python skills/generate-knowledge-qa/scripts/generate_qa_groundtruth.py 原子知识.json \
  -o qa_candidates.json --count 30 --min-ids 3 --target-ids 6 --max-ids 10
```

Rewrite workflow with explicit review fields:

```bash
python skills/generate-knowledge-qa/scripts/generate_qa_groundtruth.py 原子知识.json \
  -o qa_candidates_review.json --count 30 --include-review-fields

python skills/generate-knowledge-qa/scripts/rewrite_and_merge.py qa_candidates_review.json \
  --export-prompts -o rewrite_prompts.jsonl

python skills/generate-knowledge-qa/scripts/rewrite_and_merge.py qa_candidates_review.json \
  --merge rewritten_results.jsonl -o graphrag_local_eval_final.json

python skills/generate-knowledge-qa/scripts/validate_qa_dataset.py graphrag_local_eval_final.json
```

`groundtruth_knowledge_ids` are fixed before rewrite. The rewrite result must either return `question`, `category`, `design_intent`, and `groundtruth_answer`, or return `reject_reason`. The merged final dataset contains only the six case fields.
