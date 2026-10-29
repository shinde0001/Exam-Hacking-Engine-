"""
Unit Tests for Pattern Analysis (engine/patterns.py)
"""

import os
import tempfile
import unittest
from engine import db, patterns


class TestPatterns(unittest.TestCase):
    def test_option_position_bias(self):
        """Verify chi-squared test on biased option distribution."""
        # Heavily biased towards 'B'
        questions = [
            {"question_type": "mcq", "answer": "B"},
            {"question_type": "mcq", "answer": "B"},
            {"question_type": "mcq", "answer": "B"},
            {"question_type": "mcq", "answer": "B"},
            {"question_type": "mcq", "answer": "B"},
            {"question_type": "mcq", "answer": "A"},
            {"question_type": "mcq", "answer": "C"},
            {"question_type": "mcq", "answer": "D"},
        ]
        bias = patterns.analyze_option_bias(questions)
        self.assertEqual(bias["favored_option"], "B")
        self.assertGreater(bias["percentages"]["B"], 50.0)

    def test_difficulty_estimation(self):
        """Verify question difficulty score calculation."""
        easy_q = {
            "question_text": "Define speed",
            "question_type": "short_answer",
            "marks": 1,
            "has_math": False,
        }
        score_easy, tier_easy = patterns.compute_question_difficulty(easy_q)
        self.assertLess(score_easy, 50)
        self.assertEqual(tier_easy, "Easy")

        hard_q = {
            "question_text": "Derive the mathematical expression for the relativistic momentum and energy relation E² = p²c² + m²c⁴ with complete calculus proof",
            "question_type": "long_answer",
            "marks": 10,
            "has_math": True,
        }
        score_hard, tier_hard = patterns.compute_question_difficulty(hard_q)
        self.assertGreater(score_hard, 65)
        self.assertEqual(tier_hard, "Hard")

    def test_distractor_analysis(self):
        """Verify distractor longest option check."""
        import json
        questions = [
            {
                "question_type": "mcq",
                "answer": "C",
                "options_json": json.dumps([
                    {"label": "A", "text": "Short"},
                    {"label": "B", "text": "Brief"},
                    {"label": "C", "text": "This is by far the longest and most comprehensive option among all choices"},
                    {"label": "D", "text": "Tiny"},
                ]),
            }
        ]
        d_res = patterns.analyze_distractors(questions)
        self.assertEqual(d_res["longest_option_correct_count"], 1)
        self.assertEqual(d_res["longest_option_correct_percent"], 100.0)

    def test_master_pattern_report(self):
        """Verify full pattern report compilation."""
        temp_dir = tempfile.TemporaryDirectory()
        db_path = os.path.join(temp_dir.name, "test_engine.db")
        db.init_db(db_path)

        doc_id = db.insert_document("Exam.pdf", "/tmp/exam.pdf", "hash1", "pdf", 100, year=2024, db_path=db_path)
        db.insert_question(
            doc_id=doc_id,
            page_number=1,
            question_number="1",
            question_text="State Newton's first law of motion and explain inertia",
            question_type="short_answer",
            marks=2,
            quality_score=90,
            db_path=db_path,
        )

        report = patterns.generate_pattern_report(db_path=db_path)
        temp_dir.cleanup()

        self.assertEqual(report["total_questions"], 1)
        self.assertIn("option_bias", report)
        self.assertIn("bloom_distribution", report)
        self.assertIn("difficulty", report)

    def test_empty_corpus_pattern_report(self):
        """Verify pattern report handles zero questions safely with complete dictionary structures."""
        temp_dir = tempfile.TemporaryDirectory()
        db_path = os.path.join(temp_dir.name, "test_empty.db")
        db.init_db(db_path)

        report = patterns.generate_pattern_report(db_path=db_path)
        temp_dir.cleanup()

        self.assertEqual(report["total_questions"], 0)
        self.assertIn("difficulty", report)
        self.assertIn("percentages", report["difficulty"])
        self.assertEqual(report["difficulty"]["percentages"]["Easy"], 0.0)
        self.assertEqual(report["difficulty"]["percentages"]["Medium"], 0.0)
        self.assertEqual(report["difficulty"]["percentages"]["Hard"], 0.0)
        self.assertEqual(report["option_bias"]["percentages"]["A"], 0.0)


if __name__ == "__main__":
    unittest.main()
