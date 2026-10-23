from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import generate_qa_groundtruth as gen
import rewrite_and_merge as merge
import validate_qa_dataset as val


TEXTS = [
    "星河产品仅限已完成风险测评的个人客户申购。",
    "申购星河产品须提交有效身份证明。",
    "星河产品申购申请应在交易日15:00前提交。",
    "星河产品每笔申购金额不得低于1万元。",
    "星河产品的客户服务热线为4000000000。",
]


def valid_case(count: int = 4) -> dict:
    return {
        "query_id": "Q001",
        "category": "理财业务-申请办理",
        "question": "申购星河产品需要满足哪些客户资格、材料、办理时间和金额要求？",
        "groundtruth_knowledge_ids": [f"K{i}" for i in range(1, count + 1)],
        "groundtruth_answer": "申购人须为已完成风险测评的个人客户。申请人须提交有效身份证明。申请须在交易日15:00前提交。每笔申购金额不得低于1万元。",
        "design_intent": "测试系统能否围绕产品申购召回客户资格、材料、办理时间和金额要求。",
    }


def dataset(cases: list[dict] | None = None) -> dict:
    return {"dataset_name": "test", "source_file": "knowledge.json",
            "design_principles": ["问题、答案和至少3个有效支撑ID相互对应。"],
            "cases": [valid_case()] if cases is None else cases}


def candidate_dataset() -> dict:
    return dataset([{"query_id": "Q001", "category": "理财业务-申请办理",
                     "supporting_knowledge": [{"knowledge_id": f"K{i}", "knowledge_text": text}
                                              for i, text in enumerate(TEXTS, 1)]}])


class ValidationTests(unittest.TestCase):
    def test_id_count_has_minimum_three_and_no_upper_bound(self) -> None:
        for count in (0, 1, 2):
            self.assertTrue(val.validate_dataset(dataset([valid_case(count)])))
        # Structural checks must not impose a target count or the old cap of 10.
        for count in (3, 4, 11, 25):
            self.assertEqual([], val.validate_dataset(dataset([valid_case(count)])))

    def test_empty_and_malformed_fields_are_rejected(self) -> None:
        for changes in ({"cases": []}, {"dataset_name": None}, {"source_file": 42},
                        {"design_principles": []}, {"extra": True}):
            self.assertTrue(val.validate_dataset({**dataset(), **changes}))
        for changes in ({"question": "？"}, {"question": "如何申购?"}, {"question": "谁能购买？何时申请？"},
                        {"groundtruth_answer": ""}, {"groundtruth_knowledge_ids": [1, 2, 3]},
                        {"groundtruth_knowledge_ids": [{}, {}, {}]},
                        {"groundtruth_knowledge_ids": ["K1", "K1", "K3"]}):
            self.assertTrue(val.validate_dataset(dataset([{**valid_case(), **changes}])), changes)

    def test_original_schema_is_exact_and_key_order_is_not_required_for_reading(self) -> None:
        case = dict(reversed(list(valid_case().items())))
        self.assertEqual([], val.validate_dataset(dataset([case])))
        case["supporting_knowledge"] = []
        self.assertTrue(val.validate_dataset(dataset([case])))

    def test_complete_answers_are_not_rejected_for_having_four_sentences(self) -> None:
        self.assertEqual([], val.validate_dataset(dataset()))

    def test_known_ids_and_duplicate_questions(self) -> None:
        self.assertTrue(val.validate_dataset(dataset(), {"K1", "K2", "K3"}))
        self.assertEqual([], val.validate_dataset(dataset(), {"K1", "K2", "K3", "K4"}))
        self.assertTrue(val.validate_dataset(dataset([valid_case(), {**valid_case(), "query_id": "Q002"}])))


class MergeTests(unittest.TestCase):
    def test_one_generation_round_can_choose_a_subset_and_preserves_final_schema(self) -> None:
        candidates, chosen = candidate_dataset(), valid_case()
        result, rejected = merge.merge_rewrites(candidates, [chosen])
        self.assertEqual([], rejected)
        self.assertEqual(dataset(), result)
        self.assertEqual(val.TOP_LEVEL_KEYS, list(result))
        self.assertEqual(val.CASE_KEYS, list(result["cases"][0]))
        self.assertNotIn("K5", result["cases"][0]["groundtruth_knowledge_ids"])

    def test_three_ids_are_allowed_but_not_forced(self) -> None:
        case = valid_case(3)
        case["question"] = "申购星河产品需要什么客户资格和材料，何时提交申请？"
        case["groundtruth_answer"] = "申购人须为已完成风险测评的个人客户，提交有效身份证明，并在交易日15:00前申请。"
        result, rejected = merge.merge_rewrites(candidate_dataset(), [case])
        self.assertFalse(rejected)
        self.assertEqual(3, len(result["cases"][0]["groundtruth_knowledge_ids"]))
        result, _ = merge.merge_rewrites(candidate_dataset(), [valid_case(4)])
        self.assertEqual(4, len(result["cases"][0]["groundtruth_knowledge_ids"]))

    def test_merge_accepts_more_than_ten_real_ids(self) -> None:
        candidates = candidate_dataset()
        candidates["cases"][0]["supporting_knowledge"] = [
            {"knowledge_id": f"K{i}", "knowledge_text": f"第{i}项办理要求的原文。"} for i in range(1, 12)]
        final, rejected = merge.merge_rewrites(candidates, [valid_case(11)])
        self.assertFalse(rejected)
        self.assertEqual(11, len(final["cases"][0]["groundtruth_knowledge_ids"]))

    def test_unknown_and_too_few_ids_are_rejected_without_padding(self) -> None:
        for ids in (["K1", "K2"], ["K1", "K2", "UNKNOWN"], ["K1", "K1", "K3"]):
            final, rejected = merge.merge_rewrites(candidate_dataset(), [{**valid_case(), "groundtruth_knowledge_ids": ids}])
            self.assertFalse(final["cases"])
            self.assertEqual(1, len(rejected))

    def test_batch_results_must_be_complete_and_unambiguous(self) -> None:
        for rows in ([], [valid_case(), valid_case()], [{**valid_case(), "query_id": "Q999"}]):
            with self.assertRaises(ValueError):
                merge.merge_rewrites(candidate_dataset(), rows)
        with self.assertRaises(ValueError):
            merge.normalize_rewrites([{"query_id": "Q002", "result": valid_case()}])

    def test_wrapped_results_are_supported_without_review_metadata(self) -> None:
        final, rejected = merge.merge_rewrites(candidate_dataset(), [{"query_id": "Q001", "result": json.dumps(valid_case(), ensure_ascii=False)}])
        self.assertFalse(rejected)
        self.assertEqual(dataset(), final)

    def test_rejection_does_not_change_surviving_ids(self) -> None:
        candidates = candidate_dataset()
        second = copy.deepcopy(candidates["cases"][0])
        second["query_id"] = "Q002"
        candidates["cases"].append(second)
        final, rejected = merge.merge_rewrites(candidates, [
            {"query_id": "Q001", "reject_reason": "无法自然组成至少3条有效支撑。"},
            {**valid_case(), "query_id": "Q002"},
        ])
        self.assertEqual(1, len(rejected))
        self.assertEqual(valid_case(), final["cases"][0])

    def test_missing_duplicate_and_truncated_materials_fail(self) -> None:
        for mode in ("missing", "duplicate", "truncated"):
            candidates = candidate_dataset()
            case = candidates["cases"][0]
            if mode == "missing":
                del case["supporting_knowledge"]
            elif mode == "duplicate":
                case["supporting_knowledge"].append(case["supporting_knowledge"][0])
            else:
                case["supporting_knowledge"][0]["is_truncated"] = True
            with self.assertRaises(ValueError):
                merge.export_prompts(candidates)


class GeneratorTests(unittest.TestCase):
    def test_candidate_is_a_material_pool_without_prefilled_groundtruth(self) -> None:
        records = {f"K{i}": {"knowledge_id": f"K{i}", "knowledge_text": text} for i, text in enumerate(TEXTS, 1)}
        case = gen.make_candidate("Q001", list(records), records)
        self.assertEqual({"query_id", "category", "supporting_knowledge"}, set(case))
        self.assertEqual(5, len(case["supporting_knowledge"]))

    def test_full_text_keeps_trailing_conditions(self) -> None:
        text = "一般办理规则。" * 700 + "但未完成风险测评的客户不得申购。"
        support = gen.make_supporting_knowledge("K1", {"knowledge_text": text})
        self.assertEqual({"knowledge_id": "K1", "knowledge_text": text}, support)

    def test_duplicate_ids_and_missing_text_fail(self) -> None:
        for rows in ([{"knowledge_id": "K1", "knowledge_text": "一"}, {"id": "K1", "text": "二"}],
                     [{"knowledge_id": "K1", "source_file": "metadata_only"}], [{"text": "无编号"}]):
            with self.assertRaises(SystemExit):
                gen.index_records(rows)

    def test_tax_name_and_numeric_features_keep_meaningful_distinctions(self) -> None:
        self.assertEqual("土地增值税", gen.record_topic({"title": "土地增值税"}))
        self.assertNotEqual(gen.lexical_features("风险权重20%"), gen.lexical_features("风险权重150%"))
        self.assertTrue(gen.lexical_features("LPR 3.5%"))


class CliTests(unittest.TestCase):
    def run_script(self, name: str, *args: str | Path, success: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run([sys.executable, "-B", str(SCRIPTS / name), *map(str, args)],
                                capture_output=True, text=True, encoding="utf-8",
                                env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1"))
        if success:
            self.assertEqual(0, result.returncode, result.stderr)
        else:
            self.assertNotEqual(0, result.returncode)
        return result

    def test_optional_batch_pipeline_needs_only_one_model_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second, candidates, prompts, results, final = [root / name for name in (
                "first.json", "second.json", "candidates.json", "prompts.jsonl", "results.jsonl", "final.json")]
            records = [{"knowledge_id": f"K{i}", "source_file": "星河产品规则",
                        "title": "星河产品申购申请办理要求", "knowledge_text": text} for i, text in enumerate(TEXTS, 1)]
            first.write_text(json.dumps(records[:3], ensure_ascii=False), encoding="utf-8")
            second.write_text(json.dumps(records[3:], ensure_ascii=False), encoding="utf-8")
            self.run_script("generate_qa_groundtruth.py", first, second, "-o", candidates,
                            "--count", "1", "--pool-size", "0", "--min-coherence", "0")
            data = json.loads(candidates.read_text(encoding="utf-8"))
            self.assertEqual(5, len(data["cases"][0]["supporting_knowledge"]))
            self.run_script("rewrite_and_merge.py", candidates, "--export-prompts", "-o", prompts)
            prompt = json.loads(prompts.read_text(encoding="utf-8").strip())["prompt"]
            self.assertIn(TEXTS[4], prompt)
            self.assertNotIn("{candidate_json}", prompt)
            results.write_text(json.dumps(valid_case(), ensure_ascii=False) + "\n", encoding="utf-8")
            self.run_script("rewrite_and_merge.py", candidates, "--merge", results, "-o", final)
            self.run_script("validate_qa_dataset.py", final, "--knowledge-file", first, second)
            result = json.loads(final.read_text(encoding="utf-8"))
            self.assertEqual(valid_case(), result["cases"][0])
            self.assertEqual(val.TOP_LEVEL_KEYS, list(result))
            self.assertEqual(val.CASE_KEYS, list(result["cases"][0]))
            self.assertEqual(6, len(list(root.iterdir())))

    def test_invalid_result_does_not_create_or_overwrite_final_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates, results, final = root / "candidates.json", root / "results.jsonl", root / "final.json"
            candidates.write_text(json.dumps(candidate_dataset(), ensure_ascii=False), encoding="utf-8")
            results.write_text(json.dumps({**valid_case(), "groundtruth_knowledge_ids": ["K1", "K2"]}, ensure_ascii=False), encoding="utf-8")
            self.run_script("rewrite_and_merge.py", candidates, "--merge", results, "-o", final, success=False)
            self.assertFalse(final.exists())
            final.write_text("previous result", encoding="utf-8")
            self.run_script("rewrite_and_merge.py", candidates, "--merge", results, "-o", final, success=False)
            self.assertEqual("previous result", final.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
