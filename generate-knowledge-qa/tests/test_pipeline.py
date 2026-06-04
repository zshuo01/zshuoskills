from __future__ import annotations

import json
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


def valid_case(ids: list[str]) -> dict:
    return {
        "query_id": "Q001",
        "category": "增值税-申报缴纳",
        "question": "在增值税申报缴纳事项中，应如何确定纳税主体和申报缴纳？",
        "groundtruth_knowledge_ids": ids,
        "groundtruth_answer": "纳税主体和申报缴纳规则应按文本列明的条件确定。",
        "design_intent": "测试系统能否围绕增值税申报缴纳召回纳税主体和申报缴纳规则。",
    }


def valid_dataset(ids: list[str]) -> dict:
    return {
        "dataset_name": "test",
        "source_file": "source.json",
        "design_principles": ["固定 ids 后再写问题。"],
        "cases": [valid_case(ids)],
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


class MergeTests(unittest.TestCase):
    def candidate_dataset(self) -> dict:
        return {
            "dataset_name": "test",
            "source_file": "source.json",
            "design_principles": ["固定 ids。"],
            "cases": [
                {
                    "query_id": "Q001",
                    "category": "增值税-申报缴纳",
                    "question": "draft？",
                    "groundtruth_knowledge_ids": ["K1", "K2", "K3"],
                    "design_intent": "draft",
                    "supporting_knowledge": [{"knowledge_id": "K1"}],
                    "quality_checks": {"final_coherence_score": 1.0},
                },
                {
                    "query_id": "Q002",
                    "category": "消费税-计税规则",
                    "question": "draft？",
                    "groundtruth_knowledge_ids": ["K4", "K5", "K6"],
                    "design_intent": "draft",
                    "supporting_knowledge": [{"knowledge_id": "K4"}],
                    "quality_checks": {"final_coherence_score": 1.0},
                },
            ],
        }

    def test_rewrite_cannot_change_ids_and_reject_is_dropped(self) -> None:
        final_dataset, rejected = merge.merge_rewrites(
            self.candidate_dataset(),
            [
                {
                    "query_id": "Q001",
                    "question": "在增值税申报缴纳事项中，应如何确定纳税主体和申报缴纳？",
                    "category": "增值税-申报缴纳",
                    "design_intent": "测试系统能否围绕增值税申报缴纳召回纳税主体和申报缴纳规则。",
                    "groundtruth_answer": "纳税主体和申报缴纳规则应按文本列明的条件确定。",
                    "groundtruth_knowledge_ids": ["BAD1", "BAD2", "BAD3"],
                },
                {"query_id": "Q002", "reject_reason": "固定 ids 无法共同支撑自然问题。"},
            ],
        )
        self.assertEqual([valid_case(["K1", "K2", "K3"])["groundtruth_knowledge_ids"]], [case["groundtruth_knowledge_ids"] for case in final_dataset["cases"]])
        self.assertEqual(1, len(rejected))
        self.assertEqual(["query_id", "category", "question", "groundtruth_knowledge_ids", "groundtruth_answer", "design_intent"], list(final_dataset["cases"][0].keys()))
        final_text = json.dumps(final_dataset, ensure_ascii=False)
        self.assertNotIn("supporting_knowledge", final_text)
        self.assertNotIn("quality_checks", final_text)


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
        self.assertEqual({"knowledge_id", "knowledge_text", "preview_text", "detected_facets", "text_length"}, set(support.keys()))
        self.assertIn("knowledge_text", support)
        self.assertIsInstance(support["detected_facets"], list)
        self.assertGreater(support["text_length"], 0)

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


if __name__ == "__main__":
    unittest.main()
