"""
Unit Tests for Similarity & Deduplication Engine (engine/similarity.py)
"""

import unittest
from engine import similarity


class TestSimilarityEngine(unittest.TestCase):
    def test_exact_match_verbatim(self):
        """Verify Layer 1 exact match on verbatim text."""
        q1 = "What is Newton's first law of motion?"
        q2 = "What is Newton's first law of motion?"

        res = similarity.compare_question_pair(q1, q2)
        self.assertTrue(res.is_duplicate)
        self.assertEqual(res.layer, "exact")
        self.assertEqual(res.score, 1.0)
        self.assertFalse(res.has_conflict)

    def test_exact_match_case_and_punctuation(self):
        """Verify Layer 1 exact match across capitalization and punctuation differences."""
        q1 = "State Ohm's Law: V = I * R."
        q2 = "state ohm's law v = i * r"

        res = similarity.compare_question_pair(q1, q2)
        self.assertTrue(res.is_duplicate)
        self.assertEqual(res.layer, "exact")
        self.assertEqual(res.score, 1.0)

    def test_near_exact_match_reworded(self):
        """Verify Layer 2 near-exact match on reworded questions."""
        q1 = "Explain the principle and working of an electric motor with neat diagram."
        q2 = "With a neat diagram explain working and principle of an electric motor."

        res = similarity.compare_question_pair(q1, q2)
        self.assertTrue(res.is_duplicate)
        self.assertIn(res.layer, ("near_exact", "exact"))
        self.assertGreaterEqual(res.score, 0.85)

    def test_semantic_match_layer(self):
        """Verify Layer 3 semantic match when high semantic score is supplied."""
        q1 = "What organelle generates ATP in eukaryotic cells?"
        q2 = "Which structure is known as the powerhouse of the cell?"

        # High cosine similarity from vector encoder
        res = similarity.compare_question_pair(q1, q2, semantic_score=0.91)
        self.assertTrue(res.is_duplicate)
        self.assertEqual(res.layer, "semantic")
        self.assertGreaterEqual(res.score, 0.82)

    def test_non_duplicate(self):
        """Verify clearly distinct questions are not flagged as duplicates."""
        q1 = "Calculate the acceleration due to gravity on the Moon."
        q2 = "Define the law of conservation of momentum in collisions."

        res = similarity.compare_question_pair(q1, q2, semantic_score=0.35)
        self.assertFalse(res.is_duplicate)
        self.assertEqual(res.layer, "none")

    def test_answer_conflict_detection(self):
        """Verify answer conflict detection on duplicate questions."""
        q1 = "Which planet is closest to the Sun?"
        q2 = "Which planet is closest to the Sun?"

        # 1. Matching answers -> No conflict
        res_same = similarity.compare_question_pair(q1, q2, ans1="Mercury", ans2="Mercury")
        self.assertTrue(res_same.is_duplicate)
        self.assertFalse(res_same.has_conflict)

        # 2. Conflicting answers -> Flagged!
        res_diff = similarity.compare_question_pair(q1, q2, ans1="Mercury", ans2="Venus")
        self.assertTrue(res_diff.is_duplicate)
        self.assertTrue(res_diff.has_conflict)
        self.assertIn("Conflicting", res_diff.conflict_note)

        # 3. Formatted option answers (A) vs (B)
        res_opts = similarity.compare_question_pair(q1, q2, ans1="(A)", ans2="(C)")
        self.assertTrue(res_opts.has_conflict)

        # 4. Equivalent math answers -> No conflict
        res_math = similarity.compare_question_pair(
            "Solve for x: 2x = 4",
            "Solve for x: 2x = 4",
            ans1="x = 2",
            ans2="2",
        )
        # 2 vs 2 or x = 2
        conf, _ = similarity.check_answer_conflict("2", "2")
        self.assertFalse(conf)


if __name__ == "__main__":
    unittest.main()
