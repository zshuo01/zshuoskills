from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SKILL_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import generate_qa_groundtruth as gen  # noqa: E402
import rewrite_and_merge as merge  # noqa: E402
import validate_qa_dataset as val  # noqa: E402
import evidence_alignment as alignment  # noqa: E402


SOURCE_TEXTS = [
    "星河产品仅限已完成风险测评的个人客户申购。",
    "申购星河产品须提交有效身份证明。",
    "星河产品申购申请应在交易日15:00前提交。",
]
ANSWER_SPANS = [
    "申购人须为已完成风险测评的个人客户。",
    "申请人须提交有效身份证明，",
    "且须在交易日15:00前提交申请。",
]
QUESTION_SPANS = ["具备什么资格", "提交哪些材料", "什么时间内申请"]


def valid_case(ids: list[str]) -> dict:
    return {
        "query_id": "Q001",
        "category": "理财业务-申请办理",
        "question": "申购星河产品需要具备什么资格、提交哪些材料，并在什么时间内申请？",
        "groundtruth_knowledge_ids": ids,
        "groundtruth_answer": "".join(ANSWER_SPANS),
        "design_intent": "测试系统能否围绕产品申购召回客户资格、申请材料和办理时间。",
    }


def valid_dataset(ids: list[str]) -> dict:
    return {
        "dataset_name": "test",
        "source_file": "source.json",
        "design_principles": ["固定 ids 后再写问题。"],
        "cases": [valid_case(ids)],
    }


def candidate_dataset() -> dict:
    data = valid_dataset(["K1", "K2", "K3"])
    case = data["cases"][0]
    case["supporting_knowledge"] = [
        {"knowledge_id": f"K{index}", "knowledge_text": text}
        for index, text in enumerate(SOURCE_TEXTS, start=1)
    ]
    case["quality_checks"] = {"final_coherence_score": 0.5}
    return data


def valid_rewrite() -> dict:
    rewrite = valid_case(["K1", "K2", "K3"])
    del rewrite["groundtruth_knowledge_ids"]
    rewrite["evidence_alignment"] = [
        {"knowledge_id": f"K{index + 1}", "question_span": QUESTION_SPANS[index],
         "answer_span": ANSWER_SPANS[index], "evidence_quote": SOURCE_TEXTS[index],
         "necessity": f"该知识给出{QUESTION_SPANS[index]}所需的独立事实。"}
        for index in range(3)
    ]
    return rewrite


def valid_review(candidate: dict, rewrite: dict) -> dict:
    return {
        "query_id": candidate["query_id"],
        "input_hash": alignment.review_input_hash(candidate, rewrite),
        "verdict": "accept", "checks": {key: True for key in alignment.REVIEW_CHECKS},
        "reason": "K1限定客户资格，K2提供必需材料，K3提供办理时间；问题和答案覆盖全部事实与限制。",
    }


class ValidateDatasetTests(unittest.TestCase):
    def assert_valid(self, data: dict) -> None:
        self.assertEqual([], val.validate_dataset(data))

    def assert_invalid_contains(self, data: dict, needle: str) -> None:
        errors = val.validate_dataset(data)
        self.assertTrue(any(needle in error for error in errors), errors)

    def test_groundtruth_id_boundaries(self) -> None:
        self.assert_invalid_contains(valid_dataset(["K1", "K2"]), "groundtruth_knowledge_ids")
        self.assert_valid(valid_dataset(["K1", "K2", "K3"]))
        self.assert_valid(valid_dataset([f"K{i}" for i in range(10)]))
        self.assert_invalid_contains(valid_dataset([f"K{i}" for i in range(11)]), "groundtruth_knowledge_ids")

    def test_question_mark_validation(self) -> None:
        data = valid_dataset(["K1", "K2", "K3"])
        data["cases"][0]["question"] = "增值税如何申报?"
        self.assert_invalid_contains(data, "question")

        data = valid_dataset(["K1", "K2", "K3"])
        data["cases"][0]["question"] = "增值税如何申报？税率如何确定？"
        self.assert_invalid_contains(data, "question")

    def test_category_format_validation(self) -> None:
        data = valid_dataset(["K1", "K2", "K3"])
        data["cases"][0]["category"] = "增值税申报缴纳"
        self.assert_invalid_contains(data, "category")

    def test_answer_three_sentence_limit(self) -> None:
        data = valid_dataset(["K1", "K2", "K3"])
        data["cases"][0]["groundtruth_answer"] = "一句。二句。三句。四句。"
        self.assert_invalid_contains(data, "groundtruth_answer")

    def test_unterminated_fourth_sentence_is_counted(self) -> None:
        data = valid_dataset(["K1", "K2", "K3"])
        data["cases"][0]["groundtruth_answer"] = "一句。二句。三句。四句"
        self.assert_invalid_contains(data, "groundtruth_answer")

    def test_empty_and_malformed_content_is_rejected(self) -> None:
        for field, value in [("question", "？"), ("groundtruth_answer", ""),
                             ("groundtruth_knowledge_ids", [1, 2, 3]),
                             ("groundtruth_knowledge_ids", [{}, {}, {}]),
                             ("groundtruth_knowledge_ids", ["K1", "K1", "K3"])]:
            with self.subTest(field=field, value=value):
                data = valid_dataset(["K1", "K2", "K3"])
                data["cases"][0][field] = value
                self.assert_invalid_contains(data, field)

    def test_exact_top_level_schema_and_types(self) -> None:
        for changes in ({"unexpected": True}, {"dataset_name": None}, {"source_file": 42},
                        {"design_principles": []}, {"cases": []}):
            data = valid_dataset(["K1", "K2", "K3"])
            data.update(changes)
            self.assertTrue(val.validate_dataset(data), changes)

    def test_field_order_is_not_json_semantics(self) -> None:
        data = valid_dataset(["K1", "K2", "K3"])
        data["cases"][0] = dict(reversed(list(data["cases"][0].items())))
        self.assert_valid(data)

    def test_unknown_source_ids_and_exposed_ids(self) -> None:
        data = valid_dataset(["K1", "K2", "K3"])
        self.assertTrue(val.validate_dataset(data, {"K1", "K2"}))
        data["cases"][0]["question"] = "K1规定的申请资格是什么？"
        self.assert_invalid_contains(data, "must not expose")

    def test_duplicate_questions_and_answer_lists(self) -> None:
        data = valid_dataset(["K1", "K2", "K3"])
        data["cases"].append(dict(data["cases"][0], query_id="Q002"))
        self.assert_invalid_contains(data, "duplicate question")
        data = valid_dataset(["K1", "K2", "K3"])
        data["cases"][0]["groundtruth_answer"] = "1. 提交材料\n2. 按期办理"
        self.assert_invalid_contains(data, "prose")


class MergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = candidate_dataset()
        self.candidate = self.candidates["cases"][0]
        self.rewrite = valid_rewrite()
        self.review = valid_review(self.candidate, self.rewrite)

    def test_reviewed_merge_preserves_original_final_schema(self) -> None:
        result, rejected = merge.merge_rewrites(self.candidates, [self.rewrite], [self.review])
        self.assertEqual([], rejected)
        self.assertEqual(valid_dataset(["K1", "K2", "K3"]), result)
        self.assertEqual(val.TOP_LEVEL_KEYS, list(result))
        self.assertEqual(val.CASE_KEYS, list(result["cases"][0]))
        self.assertEqual([], val.validate_dataset(result))

    def test_changed_fixed_ids_are_rejected(self) -> None:
        self.rewrite["groundtruth_knowledge_ids"] = ["BAD1", "BAD2", "BAD3"]
        result, rejected = merge.merge_rewrites(self.candidates, [self.rewrite], [])
        self.assertEqual([], result["cases"])
        self.assertIn("fixed candidate ids", rejected[0]["reject_reason"])

    def test_reject_is_recorded(self) -> None:
        result, rejected = merge.merge_rewrites(self.candidates, [{"query_id": "Q001", "reject_reason": "证据重复，无法成题。"}], [])
        self.assertEqual([], result["cases"])
        self.assertEqual("证据重复，无法成题。", rejected[0]["reject_reason"])

    def test_rejected_candidate_does_not_change_surviving_knowledge_ids(self) -> None:
        second = copy.deepcopy(self.candidate)
        second["query_id"] = "Q002"
        second["groundtruth_knowledge_ids"] = ["K4", "K5", "K6"]
        rewrite = copy.deepcopy(self.rewrite)
        rewrite["query_id"] = "Q002"
        for index in range(3):
            second["supporting_knowledge"][index]["knowledge_id"] = f"K{index + 4}"
            rewrite["evidence_alignment"][index]["knowledge_id"] = f"K{index + 4}"
        self.candidates["cases"].append(second)
        result, rejected = merge.merge_rewrites(
            self.candidates, [{"query_id": "Q001", "reject_reason": "无法形成独立问题。"}, rewrite],
            [valid_review(second, rewrite)],
        )
        self.assertEqual(1, len(rejected))
        self.assertEqual("Q001", result["cases"][0]["query_id"])
        self.assertEqual(["K4", "K5", "K6"], result["cases"][0]["groundtruth_knowledge_ids"])
        self.assertEqual([], val.validate_dataset(result))

    def test_missing_duplicate_and_unknown_batch_results_fail(self) -> None:
        for rows in ([], [self.rewrite, self.rewrite], [dict(self.rewrite, query_id="Q999")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                merge.merge_rewrites(self.candidates, rows, [])
        for rows in ([], [self.review, self.review], [dict(self.review, query_id="Q999")]):
            with self.subTest(review_rows=rows), self.assertRaises(ValueError):
                merge.merge_rewrites(self.candidates, [self.rewrite], rows)

    def test_separate_review_can_reject_semantic_mismatch(self) -> None:
        self.review["checks"]["all_ids_necessary"] = False
        self.review["verdict"] = "reject"
        self.review["reason"] = "其中一个知识点没有独立的必要贡献。"
        result, rejected = merge.merge_rewrites(self.candidates, [self.rewrite], [self.review])
        self.assertEqual([], result["cases"])
        self.assertIn("all_ids_necessary", rejected[0]["reject_reason"])

    def test_truthy_strings_do_not_count_as_review_approval(self) -> None:
        self.review["checks"]["numbers_correct"] = "true"
        result, rejected = merge.merge_rewrites(self.candidates, [self.rewrite], [self.review])
        self.assertFalse(result["cases"])
        self.assertIn("boolean", rejected[0]["reject_reason"])

    def test_stale_review_is_not_reused_after_material_change(self) -> None:
        self.candidate["supporting_knowledge"][0]["knowledge_text"] += "其他条件另行列明。"
        with self.assertRaisesRegex(ValueError, "input_hash mismatch"):
            merge.merge_rewrites(self.candidates, [self.rewrite], [self.review])

    def test_wrapped_results_preserve_hash_and_reject_conflicting_ids(self) -> None:
        wrapped_review = {"query_id": "Q001", "input_hash": self.review["input_hash"],
                          "result": {key: value for key, value in self.review.items() if key not in ("query_id", "input_hash")}}
        result, _ = merge.merge_rewrites(self.candidates, [{"query_id": "Q001", "result": self.rewrite}], [wrapped_review])
        self.assertEqual(1, len(result["cases"]))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            merge.normalize_rewrites([{"query_id": "Q002", "result": self.rewrite}])

    def test_export_without_source_material_fails(self) -> None:
        del self.candidate["supporting_knowledge"]
        with self.assertRaisesRegex(ValueError, "include-review-fields"):
            merge.export_prompts(self.candidates)

    def test_review_prompt_binds_current_input(self) -> None:
        rows, rejected = merge.export_review_prompts(self.candidates, [self.rewrite])
        self.assertFalse(rejected)
        self.assertEqual(self.review["input_hash"], rows[0]["input_hash"])
        self.assertIn(SOURCE_TEXTS[0], rows[0]["prompt"])
        self.assertIn(self.rewrite["question"], rows[0]["prompt"])


class EvidenceAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidate = candidate_dataset()["cases"][0]
        self.rewrite = valid_rewrite()

    def test_valid_full_alignment(self) -> None:
        self.assertEqual([], alignment.alignment_errors(self.candidate, self.rewrite))

    def test_fabricated_quote_is_rejected(self) -> None:
        self.rewrite["evidence_alignment"][0]["evidence_quote"] = "任何客户均可购买。"
        self.assertTrue(any("exact source" in error for error in alignment.alignment_errors(self.candidate, self.rewrite)))

    def test_every_id_and_full_answer_must_be_mapped(self) -> None:
        self.rewrite["evidence_alignment"].pop()
        errors = alignment.alignment_errors(self.candidate, self.rewrite)
        self.assertTrue(any("no question/answer contribution" in error for error in errors))
        self.assertTrue(any("unmapped content" in error for error in errors))

    def test_question_and_answer_spans_must_actually_exist(self) -> None:
        self.rewrite["evidence_alignment"][0]["question_span"] = "销售渠道"
        self.rewrite["evidence_alignment"][1]["answer_span"] = "无需身份证明"
        errors = alignment.alignment_errors(self.candidate, self.rewrite)
        self.assertTrue(any("question_span" in error for error in errors))
        self.assertTrue(any("answer_span" in error for error in errors))

    def test_five_percent_cannot_be_rewritten_as_ninety_nine(self) -> None:
        self.candidate["supporting_knowledge"][2]["knowledge_text"] = "申购费率为5%。"
        self.rewrite["question"] = "申购星河产品需要具备什么资格、提交哪些材料，费率是多少？"
        self.rewrite["groundtruth_answer"] = "".join(ANSWER_SPANS[:2]) + "申购费率为99%。"
        row = self.rewrite["evidence_alignment"][2]
        row.update(question_span="费率是多少", answer_span="申购费率为99%。", evidence_quote="申购费率为5%。")
        errors = alignment.alignment_errors(self.candidate, self.rewrite)
        self.assertTrue(any("numbers absent" in error for error in errors), errors)

    def test_number_formatting_does_not_change_value_or_round_large_amounts(self) -> None:
        self.assertEqual(alignment.number_tokens("2026-10-23，费率５.００％"), alignment.number_tokens("2026年10月23日，费率5%"))
        self.assertNotEqual(alignment.number_tokens("123456789012345678901234567890"),
                            alignment.number_tokens("123456789012345678901234567891"))
        self.assertNotEqual(alignment.number_tokens("-5%"), alignment.number_tokens("5%"))

    def test_calculated_numbers_require_derivation_and_still_need_review(self) -> None:
        self.candidate["supporting_knowledge"][2]["knowledge_text"] = "申购金额100万元，费用80万元。"
        self.rewrite["groundtruth_answer"] = "".join(ANSWER_SPANS[:2]) + "扣费后剩余20万元。"
        row = self.rewrite["evidence_alignment"][2]
        row.update(answer_span="扣费后剩余20万元。", evidence_quote="申购金额100万元，费用80万元。")
        self.assertTrue(alignment.alignment_errors(self.candidate, self.rewrite))
        row["derivation"] = "引用金额100万元减去引用费用80万元，100-80=20万元。"
        self.assertEqual([], alignment.alignment_errors(self.candidate, self.rewrite))
        with self.assertRaises(ValueError):
            merge.merge_rewrites({**candidate_dataset(), "cases": [self.candidate]}, [self.rewrite], [])


class GeneratorTests(unittest.TestCase):
    def test_jaccard_dedupe_removes_high_overlap(self) -> None:
        clusters = [
            {"ids": ["A", "B", "C", "D"], "scores": {"final_coherence_score": 0.9}},
            {"ids": ["A", "B", "C", "D", "E"], "scores": {"final_coherence_score": 0.8}},
            {"ids": ["X", "Y", "Z"], "scores": {"final_coherence_score": 0.7}},
        ]
        deduped = gen.dedupe_clusters(clusters, overlap_threshold=0.75)
        self.assertEqual([["A", "B", "C", "D"], ["X", "Y", "Z"]], [item["ids"] for item in deduped])

    def test_supporting_knowledge_shape(self) -> None:
        record = {
            "knowledge_id": "K1",
            "title": "增值税申报",
            "knowledge_text": "增值税纳税人应当申报缴纳，纳税期限和纳税地点按规定办理。",
        }
        support = gen.make_supporting_knowledge("K1", record, max_supporting_chars=3000)
        self.assertEqual({"knowledge_id", "knowledge_text", "preview_text", "detected_facets", "text_length", "is_truncated"}, set(support.keys()))
        self.assertIn("knowledge_text", support)
        self.assertIsInstance(support["detected_facets"], list)
        self.assertGreater(support["text_length"], 0)
        self.assertFalse(support["is_truncated"])

    def test_full_text_keeps_trailing_exception_and_truncation_is_blocked(self) -> None:
        text = "一般条件。" * 700 + "但未完成风险测评的客户不得申购。"
        record = {"knowledge_id": "K1", "knowledge_text": text}
        complete = gen.make_supporting_knowledge("K1", record, max_supporting_chars=0)
        self.assertEqual(text, complete["knowledge_text"])
        truncated = gen.make_supporting_knowledge("K1", record, max_supporting_chars=3000)
        data = candidate_dataset()
        data["cases"][0]["supporting_knowledge"][0] = truncated
        with self.assertRaisesRegex(ValueError, "truncated"):
            merge.export_prompts(data)

    def test_duplicate_ids_and_missing_text_are_not_silently_discarded(self) -> None:
        for records in ([{"knowledge_id": "K1", "knowledge_text": "第一版"}, {"knowledge_id": "K1", "knowledge_text": "第二版"}],
                        [{"knowledge_id": "K1", "source_file": "metadata_only.json"}],
                        [{"knowledge_text": "没有编号"}], [42]):
            with self.subTest(records=records), self.assertRaises(SystemExit):
                gen.index_records(records)

    def test_specific_tax_name_is_not_replaced_by_nested_tax_name(self) -> None:
        self.assertEqual("土地增值税", gen.record_topic({"title": "土地增值税", "knowledge_text": "土地增值税规则。"}))

    def test_lexical_features_distinguish_numbers_and_keep_financial_acronyms(self) -> None:
        self.assertNotEqual(gen.lexical_features("风险权重为20%。"), gen.lexical_features("风险权重为150%。"))
        self.assertTrue(gen.lexical_features("LPR 3.5%"))

    def test_fallback_topic_is_reproducible_between_processes(self) -> None:
        probe = "import sys;sys.path.insert(0,sys.argv[1]);import generate_qa_groundtruth as g;print(ascii(g.infer_main_topic([{'knowledge_text':'银行账户开立需要审核身份资料。'}]*3)))"
        topics = set()
        for seed in ("1", "2", "3"):
            proc = subprocess.run([sys.executable, "-B", "-c", probe, str(SCRIPTS_DIR)],
                                  env=dict(os.environ, PYTHONHASHSEED=seed), capture_output=True, text=True, check=True)
            topics.add(proc.stdout.strip())
        self.assertEqual(1, len(topics))

    def test_cli_basic_runs(self) -> None:
        records = []
        for idx in range(1, 7):
            records.append(
                {
                    "knowledge_id": f"K{idx:03d}",
                    "source_file": "增值税规则",
                    "title": "增值税申报缴纳",
                    "knowledge_text": f"增值税纳税人应当申报缴纳增值税，纳税期限、纳税地点和计税依据按规定办理。第{idx}条补充申报缴纳要求。",
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "knowledge.json"
            output_path = tmp_path / "qa_candidates.json"
            input_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "generate_qa_groundtruth.py"),
                    str(input_path),
                    "-o",
                    str(output_path),
                    "--count",
                    "1",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, proc.returncode, proc.stderr)
            data = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(1, len(data["cases"]))
            self.assertEqual(["query_id", "category", "question", "groundtruth_knowledge_ids", "groundtruth_answer", "design_intent"], list(data["cases"][0].keys()))
            self.assertEqual("", data["cases"][0]["question"])
            self.assertEqual("", data["cases"][0]["groundtruth_answer"])
            self.assertEqual("", data["cases"][0]["design_intent"])

    def test_cli_include_review_fields(self) -> None:
        records = []
        for idx in range(1, 7):
            records.append(
                {
                    "knowledge_id": f"K{idx:03d}",
                    "source_file": "增值税规则",
                    "title": "增值税申报缴纳",
                    "knowledge_text": f"增值税纳税人应当申报缴纳增值税，纳税期限、纳税地点和计税依据按规定办理。第{idx}条补充申报缴纳要求。",
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "knowledge.json"
            output_path = tmp_path / "qa_candidates.json"
            input_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "generate_qa_groundtruth.py"),
                    str(input_path),
                    "-o",
                    str(output_path),
                    "--count",
                    "1",
                    "--include-review-fields",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, proc.returncode, proc.stderr)
            data = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("supporting_knowledge", data["cases"][0])
            self.assertIn("quality_checks", data["cases"][0])


class PipelineCliTests(unittest.TestCase):
    def run_script(self, script: str, *args: str | Path, success: bool = True) -> subprocess.CompletedProcess:
        proc = subprocess.run([sys.executable, "-B", str(SCRIPTS_DIR / script), *map(str, args)],
                              capture_output=True, text=True, encoding="utf-8",
                              env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1"))
        if success:
            self.assertEqual(0, proc.returncode, proc.stderr)
        else:
            self.assertNotEqual(0, proc.returncode, proc.stdout)
        return proc

    def write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def test_complete_cli_pipeline_keeps_exact_final_json_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source, candidates_path = folder / "knowledge.json", folder / "candidates.json"
            rewrite_path, review_path = folder / "rewrites.jsonl", folder / "reviews.jsonl"
            rewrite_prompts, review_prompts = folder / "rewrite_prompts.jsonl", folder / "review_prompts.jsonl"
            final_path = folder / "final.json"
            source.write_text(json.dumps([
                {"knowledge_id": f"K{index}", "source_file": "星河产品",
                 "title": "星河产品申购申请办理规则", "knowledge_text": text}
                for index, text in enumerate(SOURCE_TEXTS, start=1)
            ], ensure_ascii=False), encoding="utf-8")
            self.run_script("generate_qa_groundtruth.py", source, "-o", candidates_path,
                            "--include-review-fields", "--count", "1", "--target-ids", "3", "--min-coherence", "0")
            self.run_script("rewrite_and_merge.py", candidates_path, "--export-prompts", "-o", rewrite_prompts)
            exported = json.loads(rewrite_prompts.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn('"query_id": "Q001"', exported["prompt"])
            self.assertNotIn("{query_id_json}", exported["prompt"])
            rewrite = valid_rewrite()
            self.write_jsonl(rewrite_path, [rewrite])
            self.run_script("rewrite_and_merge.py", candidates_path, "--export-review-prompts", rewrite_path, "-o", review_prompts)
            review_row = json.loads(review_prompts.read_text(encoding="utf-8").splitlines()[0])
            candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
            review = valid_review(candidates["cases"][0], rewrite)
            self.assertEqual(review["input_hash"], review_row["input_hash"])
            self.write_jsonl(review_path, [review])
            self.run_script("rewrite_and_merge.py", candidates_path, "--merge", rewrite_path, "--reviews", review_path, "-o", final_path)
            self.run_script("validate_qa_dataset.py", final_path, "--knowledge-file", source)
            final = json.loads(final_path.read_text(encoding="utf-8"))
            self.assertEqual(val.TOP_LEVEL_KEYS, list(final))
            self.assertEqual(valid_case(["K1", "K2", "K3"]), final["cases"][0])
            self.assertEqual(val.CASE_KEYS, list(final["cases"][0]))

    def test_failed_or_missing_review_cannot_write_a_final_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            candidate_path, rewrite_path, review_path = folder / "candidates.json", folder / "rewrites.jsonl", folder / "reviews.jsonl"
            final_path = folder / "final.json"
            data, rewrite = candidate_dataset(), valid_rewrite()
            candidate_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            self.write_jsonl(rewrite_path, [rewrite])
            self.run_script("rewrite_and_merge.py", candidate_path, "--merge", rewrite_path, "-o", final_path, success=False)
            self.assertFalse(final_path.exists())
            review = valid_review(data["cases"][0], rewrite)
            review["checks"]["answer_supported"] = False
            review["verdict"] = "reject"
            self.write_jsonl(review_path, [review])
            self.run_script("rewrite_and_merge.py", candidate_path, "--merge", rewrite_path, "--reviews", review_path, "-o", final_path, success=False)
            self.assertFalse(final_path.exists())
            self.assertTrue(final_path.with_suffix(".json.rejected_cases.json").exists())
            final_path.write_text("previous result", encoding="utf-8")
            self.run_script("rewrite_and_merge.py", candidate_path, "--merge", rewrite_path, "--reviews", review_path, "-o", final_path, success=False)
            self.assertEqual("previous result", final_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
