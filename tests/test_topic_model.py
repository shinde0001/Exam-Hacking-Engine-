"""
Unit Tests for Topic Modeling (engine/topic_model.py)
"""

import unittest
from engine import topic_model


class TestTopicModel(unittest.TestCase):
    def test_run_topic_modeling(self):
        """Verify NMF topic discovery on distinct question clusters."""
        questions = [
            {"id": 1, "question_text": "Calculate kinetic energy and velocity of moving particle"},
            {"id": 2, "question_text": "Calculate work done and power in kinetic mechanics"},
            {"id": 3, "question_text": "State first law of thermodynamics heat and internal energy"},
            {"id": 4, "question_text": "Explain Carnot engine efficiency in thermodynamics"},
        ]

        result = topic_model.run_topic_modeling(questions, num_topics=2, method="nmf")
        self.assertGreater(len(result.topics), 0)
        self.assertEqual(len(result.assignments), 4)

        # Ensure topics have keywords
        for t in result.topics:
            self.assertGreater(len(t.top_keywords), 0)
            self.assertIsInstance(t.name, str)

    def test_fallback_on_single_question(self):
        """Verify graceful fallback for minimal input."""
        questions = [{"id": 1, "question_text": "Single question test"}]
        result = topic_model.run_topic_modeling(questions)
        self.assertEqual(len(result.topics), 1)
        self.assertEqual(len(result.assignments), 1)
        self.assertEqual(result.assignments[0].topic_id, 0)


if __name__ == "__main__":
    unittest.main()
