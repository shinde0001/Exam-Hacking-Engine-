"""
Unit Tests for Question Validator (engine/validator.py)
"""

import unittest
import config
from engine.extractor import ExtractedQuestion
from engine import validator


class TestValidator(unittest.TestCase):
    def test_short_question_failure(self):
        """Verify questions < MIN_QUESTION_LENGTH fail quality check."""
        q = ExtractedQuestion(
            question_number="1",
            question_text="Short?",  # 6 chars, < 10
            question_type="short_answer",
        )
        res = validator.validate_question(q)
        self.assertFalse(res.is_valid)
        self.assertLess(res.quality_score, config.QUALITY_SCORE_THRESHOLD)
        self.assertTrue(any("too short" in issue for issue in res.issues))

    def test_valid_mcq_scoring(self):
        """Verify well-formed 4-option MCQ achieves high quality score."""
        q = ExtractedQuestion(
            question_number="1",
            question_text="What is the chemical symbol for Gold in the periodic table?",
            question_type="mcq",
            options=[
                {"label": "A", "text": "Ag"},
                {"label": "B", "text": "Au"},
                {"label": "C", "text": "Fe"},
                {"label": "D", "text": "Pb"},
            ],
            answer="B",
        )
        res = validator.validate_question(q)
        self.assertTrue(res.is_valid)
        self.assertGreaterEqual(res.quality_score, 80)
        self.assertEqual(len(res.issues), 0)

    def test_mcq_missing_options_penalty(self):
        """Verify MCQ with missing options receives penalty."""
        q = ExtractedQuestion(
            question_number="2",
            question_text="Which of the following are primary colors of light?",
            question_type="mcq",
            options=[
                {"label": "A", "text": "Red"},
                {"label": "B", "text": "Blue"},
            ],  # Only 2 options
        )
        res = validator.validate_question(q)
        self.assertTrue(any("options" in issue for issue in res.issues))

    def test_ocr_artifact_detection(self):
        """Verify questions with heavy OCR noise are penalized."""
        q = ExtractedQuestion(
            question_number="3",
            question_text="Solve the ¬§ following | || equation for ¬ value § of x",
            question_type="short_answer",
        )
        res = validator.validate_question(q)
        self.assertTrue(any("OCR" in issue for issue in res.issues))


if __name__ == "__main__":
    unittest.main()
