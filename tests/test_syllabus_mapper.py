"""
Unit Tests for Syllabus Mapper (engine/syllabus_mapper.py)
"""

import unittest
from engine import syllabus_mapper


class TestSyllabusMapper(unittest.TestCase):
    def test_parse_syllabus_markdown(self):
        """Verify parsing markdown / text syllabus."""
        text = """
        Chapter 1: Mechanics and Kinematics
        - Velocity, acceleration, projectile motion

        Chapter 2: Thermodynamics
        - Heat, first law, Carnot cycle
        """
        chapters = syllabus_mapper.parse_syllabus_text(text)
        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0].chapter_number, 1)
        self.assertIn("Mechanics", chapters[0].title)
        self.assertEqual(chapters[1].chapter_number, 2)
        self.assertIn("Thermodynamics", chapters[1].title)

    def test_map_questions_to_syllabus(self):
        """Verify question mapping and marks weightage calculation."""
        syllabus_text = """
        Chapter 1: Mechanics
        - velocity, acceleration, force

        Chapter 2: Thermodynamics
        - heat, temperature, entropy
        """
        chapters = syllabus_mapper.parse_syllabus_text(syllabus_text)

        questions = [
            {"id": 1, "question_text": "Calculate the velocity and force of the particle", "marks": 5},
            {"id": 2, "question_text": "State first law of thermodynamics regarding heat and temperature", "marks": 5},
        ]

        weightages = syllabus_mapper.map_questions_to_syllabus(questions, chapters)
        self.assertEqual(len(weightages), 2)

        # Total marks is 10, each should have 5 marks (50%)
        ch1 = next(c for c in weightages if c.chapter_number == 1)
        ch2 = next(c for c in weightages if c.chapter_number == 2)

        self.assertEqual(ch1.total_marks, 5)
        self.assertEqual(ch1.weightage_percent, 50.0)
        self.assertEqual(ch2.total_marks, 5)
        self.assertEqual(ch2.weightage_percent, 50.0)


if __name__ == "__main__":
    unittest.main()
