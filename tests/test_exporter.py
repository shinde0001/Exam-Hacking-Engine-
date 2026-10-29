"""
Unit Tests for Exporter Layer (engine/exporter.py)
Tests CSV export, Anki flashcard deck generation, Study Plan CSV export,
and ReportLab PDF mock examination paper generation.
"""

import os
import tempfile
import unittest

from engine import exporter


class TestExporter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sample_questions = [
            {
                "id": 1,
                "question_number": "1",
                "question_type": "mcq",
                "marks": 1,
                "topic_name": "Mechanics",
                "bloom_level": "remember",
                "difficulty_tier": "Easy",
                "composite_score": 0.88,
                "confidence_badge": "🔴",
                "confidence_label": "Very High Confidence",
                "question_text": "What is the formula for kinetic energy?",
                "options": [
                    {"label": "A", "text": "KE = 1/2 * m * v^2"},
                    {"label": "B", "text": "KE = m * g * h"},
                    {"label": "C", "text": "KE = F * d"},
                    {"label": "D", "text": "KE = p * v"},
                ],
                "answer": "A",
            },
            {
                "id": 2,
                "question_number": "2",
                "question_type": "long_answer",
                "marks": 5,
                "topic_name": "Electromagnetism",
                "bloom_level": "apply",
                "difficulty_tier": "Hard",
                "composite_score": 0.75,
                "confidence_badge": "🟡",
                "confidence_label": "Medium Confidence",
                "question_text": "Derive Maxwell's equations in vacuum and explain displacement current.",
                "options": [],
                "answer": "Displacement current ID = epsilon0 * dPhi_E / dt",
            },
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_export_questions_to_csv(self):
        """Verify questions export produces valid CSV with headers and row values."""
        csv_str = exporter.export_questions_to_csv(self.sample_questions)
        self.assertIn("Question ID,Number,Type,Marks", csv_str)
        self.assertIn("What is the formula for kinetic energy?", csv_str)
        self.assertIn("Derive Maxwell's equations", csv_str)
        self.assertIn("KE = 1/2 * m * v^2", csv_str)

    def test_export_questions_to_anki(self):
        """Verify Anki export produces tab-delimited cards with HTML styling and tags."""
        anki_str = exporter.export_questions_to_anki(self.sample_questions)
        lines = [line for line in anki_str.strip().split("\n") if line]
        self.assertEqual(len(lines), 2)

        # First card (MCQ)
        mcq_card = lines[0].split("\t")
        self.assertEqual(len(mcq_card), 3)  # Front, Back, Tags
        front, back, tags = mcq_card
        self.assertIn("kinetic energy", front)
        self.assertIn("(A)", front)
        self.assertIn("Correct Answer:", back)
        self.assertIn("ExamEngine", tags)
        self.assertIn("Mechanics", tags)

    def test_export_study_plan_to_csv(self):
        """Verify day-by-day study schedule export produces valid CSV."""
        daily_schedule = [
            {
                "day_number": 1,
                "date_str": "2026-11-01",
                "tasks": [
                    {
                        "task_type": "learn",
                        "estimated_minutes": 20,
                        "question_number": "1",
                        "question_type": "mcq",
                        "marks": 1,
                        "topic_name": "Mechanics",
                        "status": "not_started",
                        "question_text": "What is kinetic energy?",
                    }
                ],
            }
        ]
        csv_str = exporter.export_study_plan_to_csv(daily_schedule)
        self.assertIn("Day,Date,Task Type,Estimated Minutes", csv_str)
        self.assertIn("2026-11-01", csv_str)
        self.assertIn("Mechanics", csv_str)

    def test_generate_mock_paper_pdf(self):
        """Verify publication-ready PDF mock examination paper creation via ReportLab."""
        mock_paper_dict = {
            "title": "Physics Predicted Final Examination 2026",
            "total_marks": 50,
            "estimated_time_minutes": 90,
            "sections": [
                {
                    "section_name": "Section A — Multiple Choice Questions",
                    "total_marks": 10,
                    "questions": [self.sample_questions[0]],
                },
                {
                    "section_name": "Section B — Descriptive Long Answer Questions",
                    "total_marks": 40,
                    "questions": [self.sample_questions[1]],
                },
            ],
        }
        pdf_filepath = os.path.join(self.temp_dir.name, "physics_mock_exam.pdf")
        out_path = exporter.generate_mock_paper_pdf(mock_paper_dict, pdf_filepath)

        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 1500)
        # Check PDF header bytes (%PDF)
        with open(out_path, "rb") as f:
            header = f.read(5)
            self.assertEqual(header, b"%PDF-")


if __name__ == "__main__":
    unittest.main()
