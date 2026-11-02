"""学習を実行せず、評価問題の分離と採点ルールの重要な境界を検証する。"""
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("glossary_experiment", ROOT / "scripts/glossary_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class ExperimentTests(unittest.TestCase):
    def test_search_query_uses_question_text_without_gold(self):
        terms = json.loads((ROOT / "data/glossary.json").read_text(encoding="utf-8"))["terms"]
        self.assertEqual(experiment.extract_search_query("講座用語『LoRA』の意味を説明してください。", terms), "LoRA")
        self.assertEqual(experiment.extract_search_query("Low-Rank Adaptationの仕組みを教えてください。", terms), "Low-Rank Adaptation")
        self.assertEqual(experiment.extract_search_query("講座用語『架空の言葉』の意味を説明してください。", terms), "架空の言葉")

    def test_reference_answers_satisfy_the_frozen_rubric(self):
        gold = experiment.read_jsonl(ROOT / "lectures/02/advanced/evaluation_gold.jsonl")
        for row in gold:
            with self.subTest(question=row["id"]):
                self.assertTrue(experiment.score_answer(row["reference"], row)["keyword_content_ok"])

    def test_question_split_and_unknown_terms_are_independent(self):
        out = ROOT / "lectures/02/advanced"
        train = experiment.read_jsonl(out / "train_messages.jsonl")
        valid = experiment.read_jsonl(out / "valid_messages.jsonl")
        gold = experiment.read_jsonl(out / "evaluation_gold.jsonl")
        tq = {r["messages"][-2]["content"] for r in train}
        vq = {r["messages"][-2]["content"] for r in valid}
        eq = {r["question"] for r in gold}
        self.assertFalse(tq & vq or tq & eq or vq & eq)
        self.assertEqual((len(train), len(valid), len(gold)), (132, 58, 35))
        train_questions = "\n".join(tq)
        for row in gold:
            if row["expected_abstention"]:
                self.assertNotIn(row["term"], train_questions)

    def test_unknown_answer_needs_explicit_source_scope(self):
        gold = {"expected_abstention": True, "concept_groups": []}
        self.assertTrue(experiment.score_answer(experiment.ABSTENTION, gold)["keyword_content_ok"])
        self.assertFalse(experiment.score_answer("これは便利な新技術です。", gold)["keyword_content_ok"])
        self.assertFalse(experiment.score_answer("", gold)["keyword_content_ok"])

    def test_known_abstention_and_partial_definition_do_not_pass(self):
        gold = {"expected_abstention": False, "concept_groups": [["固定"], ["行列", "低ランク"]]}
        self.assertTrue(experiment.score_answer("元の重みを固定して小さな行列を学習します。", gold)["keyword_content_ok"])
        self.assertFalse(experiment.score_answer("元の重みを固定します。", gold)["keyword_content_ok"])
        self.assertFalse(experiment.score_answer(experiment.ABSTENTION, gold)["keyword_content_ok"])


if __name__ == "__main__":
    unittest.main()
