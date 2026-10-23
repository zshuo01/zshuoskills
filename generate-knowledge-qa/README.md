# Generate Knowledge QA

Generate Chinese financial GraphRAG local-query evaluation cases whose
questions, answers, and fixed knowledge ids match. See [SKILL.md](SKILL.md) for
the full contract and workflow.

The final deliverable remains a `.json` file with exactly:

- Top level: `dataset_name`, `source_file`, `design_principles`, `cases`.
- Each case: `query_id`, `category`, `question`, `groundtruth_knowledge_ids`,
  `groundtruth_answer`, `design_intent`.

Review data never enter that file. Each case still uses 3–10 fixed ids and an
answer of at most three sentences. Graph structure is not an output or review
requirement.

## Pipeline

Run these commands from the repository root; for an installed copy, replace
`generate-knowledge-qa` with that skill folder’s path.

```bash
python generate-knowledge-qa/scripts/generate_qa_groundtruth.py knowledge.json -o qa_candidates_review.json --count 30 --include-review-fields
python generate-knowledge-qa/scripts/rewrite_and_merge.py qa_candidates_review.json --export-prompts -o rewrite_prompts.jsonl
```

Process each rewrite prompt with an LLM and save `rewritten_results.jsonl`.
Successful results include an intermediate `evidence_alignment`: each id’s
exact quote, question span, answer span, and necessary contribution. Unsuitable
sets return `reject_reason`. Then export a fresh review pass:

```bash
python generate-knowledge-qa/scripts/rewrite_and_merge.py qa_candidates_review.json --export-review-prompts rewritten_results.jsonl -o alignment_review_prompts.jsonl
```

Process these prompts and save `reviewed_results.jsonl`, retaining the supplied
`query_id` and `input_hash`. A review checks question answerability, answer
support, each id’s necessity, scope/conditions, and numbers. Then merge:

```bash
python generate-knowledge-qa/scripts/rewrite_and_merge.py qa_candidates_review.json --merge rewritten_results.jsonl --reviews reviewed_results.jsonl -o graphrag_local_eval_final.json
python generate-knowledge-qa/scripts/validate_qa_dataset.py graphrag_local_eval_final.json --knowledge-file knowledge.json
```

Source texts are complete by default. Missing/truncated supports, changed fixed
ids, fabricated quotes, unmapped answer text, and missing/duplicate batch
results are blocked. Direct numbers absent from their citations require an
explicit calculation derivation and subsequent review. Invalid rewrites and
rejected reviews are reported separately; a failed or empty final dataset is
not written. If an output already exists, failure leaves it untouched; use a
new output path for a new run to avoid confusing it with an earlier dataset.

These checks establish traceability; semantic correctness still depends on the
actual review and a human spot-check. They do not claim to prove entailment.

Run the regression tests with:

```bash
python -B -m unittest discover -s generate-knowledge-qa/tests -v
```
