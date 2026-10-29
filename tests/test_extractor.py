"""
Unit Tests for Question Extractor (engine/extractor.py)
"""

import os
import tempfile
import unittest
from engine import db, extractor


class TestExtractor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_engine.db")
        db.init_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extract_marks(self):
        """Verify marks extraction and stripping."""
        marks, cleaned = extractor.extract_marks("Explain the theory of relativity. [5 Marks]")
        self.assertEqual(marks, 5)
        self.assertEqual(cleaned, "Explain the theory of relativity.")

        marks2, cleaned2 = extractor.extract_marks("What is an atom? (2 marks)")
        self.assertEqual(marks2, 2)
        self.assertEqual(cleaned2, "What is an atom?")

        marks3, _ = extractor.extract_marks("State Ohm's law.")
        self.assertIsNone(marks3)

    def test_extract_answer(self):
        """Verify answer key detection."""
        ans, cleaned = extractor.extract_answer("Which planet is closest to the Sun? Ans: Mercury")
        self.assertEqual(ans, "Mercury")
        self.assertEqual(cleaned, "Which planet is closest to the Sun?")

        ans2, cleaned2 = extractor.extract_answer("Find the square root of 64. Answer: (B)")
        self.assertEqual(ans2, "B")

    def test_extract_mcq_options(self):
        """Verify extraction of options (A), (B), (C), (D)."""
        text = (
            "What is the SI unit of electric current?\n"
            "(A) Volt\n"
            "(B) Ampere\n"
            "(C) Ohm\n"
            "(D) Watt"
        )
        options, stem = extractor.extract_mcq_options(text)
        self.assertEqual(len(options), 4)
        self.assertEqual(options[0]["label"], "A")
        self.assertEqual(options[0]["text"], "Volt")
        self.assertEqual(options[1]["label"], "B")
        self.assertEqual(options[1]["text"], "Ampere")
        self.assertEqual(stem, "What is the SI unit of electric current?")

    def test_question_type_classification(self):
        """Verify auto-classification of question types."""
        # MCQ
        self.assertEqual(
            extractor.classify_question_type("Sample question", has_options=True),
            "mcq",
        )
        # Numerical
        self.assertEqual(
            extractor.classify_question_type(
                "Calculate the velocity of a 2kg mass moving with 100J kinetic energy",
                has_options=False,
            ),
            "numerical",
        )
        # Short answer
        self.assertEqual(
            extractor.classify_question_type("Define acceleration", has_options=False, marks=2),
            "short_answer",
        )
        # Long answer
        self.assertEqual(
            extractor.classify_question_type(
                "Derive the lens maker's formula with a neat ray diagram",
                has_options=False,
                marks=5,
            ),
            "long_answer",
        )

    def test_split_and_parse_question_blocks(self):
        """Verify parsing multiple questions in continuous exam text."""
        raw_paper = (
            "SECTION A: ALL QUESTIONS COMPULSORY\n\n"
            "Question 1. What is the powerhouse of the cell? [1 Mark]\n"
            "(A) Nucleus\n"
            "(B) Mitochondria\n"
            "(C) Ribosome\n"
            "(D) Chloroplast\n"
            "Ans: B\n\n"
            "2. Calculate the work done in moving a charge of 2C across a 12V potential difference.\n\n"
            "Q3. Explain the mechanism of photosynthesis in detail with relevant chemical equations. [5 Marks]\n"
        )

        blocks = extractor.split_text_into_question_blocks(raw_paper)
        self.assertEqual(len(blocks), 3)

        q1 = extractor.parse_question_block(blocks[0][0], blocks[0][1], page_number=1)
        self.assertEqual(q1.question_number, "1")
        self.assertEqual(q1.question_type, "mcq")
        self.assertEqual(len(q1.options), 4)
        self.assertEqual(q1.answer, "B")
        self.assertEqual(q1.marks, 1)

        q2 = extractor.parse_question_block(blocks[1][0], blocks[1][1], page_number=1)
        self.assertEqual(q2.question_number, "2")
        self.assertEqual(q2.question_type, "numerical")

        q3 = extractor.parse_question_block(blocks[2][0], blocks[2][1], page_number=1)
        self.assertEqual(q3.question_number, "3")
        self.assertEqual(q3.question_type, "long_answer")
        self.assertEqual(q3.marks, 5)

    def test_extract_and_store_pipeline(self):
        """Verify full document extraction pipeline writes to DB."""
        doc_id = db.insert_document(
            filename="Test_Exam_2024.txt",
            filepath="/tmp/test.txt",
            file_hash="mockhash2024",
            file_type="txt",
            file_size=200,
            db_path=self.db_path,
        )
        db.insert_document_page(
            doc_id=doc_id,
            page_number=1,
            raw_text="1. State Newton's second law of motion. [2 marks]\n\n2. Calculate the force required to accelerate a 5kg mass at 2m/s². [3 marks]",
            char_count=130,
            extraction_method="text_read",
            db_path=self.db_path,
        )

        stored = extractor.extract_and_store_document_questions(doc_id, db_path=self.db_path)
        self.assertEqual(len(stored), 2)

        # Check DB records
        doc = db.get_document_by_id(doc_id, db_path=self.db_path)
        self.assertEqual(doc["status"], "extracted")

        db_questions = db.get_questions_by_document(doc_id, db_path=self.db_path)
        self.assertEqual(len(db_questions), 2)
        self.assertEqual(db_questions[0]["question_number"], "1")
        self.assertEqual(db_questions[0]["marks"], 2)


if __name__ == "__main__":
    unittest.main()
