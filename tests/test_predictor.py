"""
Unit Tests for Phase 5: Prediction Engine (engine/predictor.py)
Tests all 6 individual algorithms, ensemble combination, confidence badges,
mock paper generation, and minimum study set cover.
"""

import os
import tempfile
import unittest

from engine import db, predictor


class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_predictor.db")
        db.init_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_bayesian_probability(self):
        """Verify Bayesian probability gives higher probability to frequently/recently seen topics."""
        documents = [
            {"id": 1, "year": 2020},
            {"id": 2, "year": 2022},
            {"id": 3, "year": 2024},
        ]
        # Topic 1 appeared in 2022 and 2024 (recent)
        # Topic 2 appeared only in 2020 (old)
        topic_doc_map = {
            1: {2, 3},
            2: {1},
        }
        probs = predictor.compute_bayesian_probabilities(topic_doc_map, documents)
        self.assertIn(1, probs)
        self.assertIn(2, probs)
        self.assertGreater(probs[1], probs[2])
        self.assertGreater(probs[1], 0.5)

    def test_markov_chain_transitions(self):
        """Verify Markov chain transition captures year-over-year topic rotation."""
        documents = [
            {"id": 1, "year": 2022},
            {"id": 2, "year": 2023},
            {"id": 3, "year": 2024},
        ]
        # In 2022: Topic 1
        # In 2023: Topic 2 (1 -> 2)
        # In 2024: Topic 1 (2 -> 1)
        # Current state is Topic 1 (2024) -> Next state should transition to Topic 2!
        topic_doc_map = {
            1: {1, 3},
            2: {2},
        }
        markov_scores = predictor.compute_markov_predictions(topic_doc_map, documents, [1, 2])
        self.assertIn(1, markov_scores)
        self.assertIn(2, markov_scores)
        # Topic 2 should have a higher transition score from Topic 1
        self.assertGreater(markov_scores[2], markov_scores[1])

    def test_poisson_arrival_rate(self):
        """Verify Poisson process probability calculation."""
        # 10 appearances across 5 exams => lambda = 2.0 => P(>=1) = 1 - e^-2 ~= 0.864
        counts = {1: 10, 2: 1}
        poisson_probs = predictor.compute_poisson_probabilities(counts, total_exam_periods=5)
        self.assertIn(1, poisson_probs)
        self.assertIn(2, poisson_probs)
        self.assertGreater(poisson_probs[1], 0.80)
        self.assertLess(poisson_probs[2], 0.50)

    def test_cyclic_pattern_detection(self):
        """Verify cyclic pattern detector identifies due and overdue topics."""
        documents = [
            {"id": 1, "year": 2018},
            {"id": 2, "year": 2020},
            {"id": 3, "year": 2022},
            {"id": 4, "year": 2024},
        ]
        # Topic 1 appeared in 2018, 2020, 2022 (gap = 2 years, regular).
        # In 2024 (2 years after 2022), it is DUE!
        topic_doc_map = {
            1: {1, 2, 3},
            2: {1},  # Single appearance -> no cycle
        }
        scores, alerts = predictor.compute_cyclic_scores(topic_doc_map, documents)
        self.assertGreater(scores[1], 0.90)
        self.assertTrue(any(a["topic_id"] == 1 and a["status"] == "DUE_NOW" for a in alerts))

    def test_association_rule_boost(self):
        """Verify association rule mining boosts co-occurring topics."""
        # Topic 1 & Topic 2 always appear together in Doc 1, Doc 2, Doc 3
        # Topic 3 appears isolated in Doc 4
        topic_doc_map = {
            1: {1, 2, 3},
            2: {1, 2, 3},
            3: {4},
        }
        priors = {1: 0.9, 2: 0.3, 3: 0.2}
        boosts = predictor.compute_association_boosts(topic_doc_map, priors, total_docs=4)
        # Topic 2 should be boosted by Topic 1's high prior
        self.assertGreater(boosts[2], 0.60)

    def test_pagerank_centrality(self):
        """Verify graph PageRank assigns higher centrality to well-connected questions."""
        questions = [
            {"id": 101, "question_text": "Central Core Concept Question"},
            {"id": 102, "question_text": "Peripheral Question A"},
            {"id": 103, "question_text": "Peripheral Question B"},
        ]
        # Question 101 connected to both 102 and 103
        edges = [
            {"source_question_id": 101, "target_question_id": 102, "score": 0.9},
            {"source_question_id": 101, "target_question_id": 103, "score": 0.9},
        ]
        pr = predictor.compute_pagerank_scores(questions, edges)
        self.assertGreater(pr[101], pr[102])
        self.assertGreater(pr[101], pr[103])

    def test_full_prediction_pipeline_and_confidence(self):
        """Test full prediction pipeline end-to-end with DB data and confidence labeling."""
        doc1 = db.insert_document("Exam2022.pdf", "/tmp/e22.pdf", "h1", "pdf", 100, year=2022, db_path=self.db_path)
        doc2 = db.insert_document("Exam2024.pdf", "/tmp/e24.pdf", "h2", "pdf", 100, year=2024, db_path=self.db_path)

        q1 = db.insert_question(
            doc_id=doc1,
            page_number=1,
            question_number="1",
            question_text="Explain Newton's laws of motion with equations and everyday examples",
            question_type="long_answer",
            marks=5,
            quality_score=95,
            db_path=self.db_path,
        )
        q2 = db.insert_question(
            doc_id=doc2,
            page_number=1,
            question_number="2",
            question_text="Calculate the velocity of a particle under constant acceleration",
            question_type="numerical",
            marks=3,
            quality_score=90,
            db_path=self.db_path,
        )

        report = predictor.run_prediction_pipeline(db_path=self.db_path)
        self.assertEqual(report.total_questions_analyzed, 2)
        self.assertEqual(len(report.ranked_questions), 2)
        self.assertIn("bayesian", report.ranked_questions[0]["algorithm_scores"])
        self.assertIn(report.ranked_questions[0]["confidence_badge"], ["🔴", "🟡", "🟢"])

    def test_mock_paper_generation(self):
        """Verify predicted mock paper generates balanced sections with total marks."""
        sample_questions = [
            {
                "id": 1,
                "question_text": "What is momentum?",
                "question_type": "mcq",
                "options": [{"label": "A", "text": "p=mv"}, {"label": "B", "text": "p=ma"}],
                "marks": 1,
                "topic_name": "Mechanics",
                "composite_score": 0.85,
            },
            {
                "id": 2,
                "question_text": "Define work done and state its SI unit.",
                "question_type": "short_answer",
                "marks": 2,
                "topic_name": "Energy",
                "composite_score": 0.80,
            },
            {
                "id": 3,
                "question_text": "Derive the work-energy theorem for a variable force.",
                "question_type": "long_answer",
                "marks": 5,
                "topic_name": "Energy",
                "composite_score": 0.78,
            },
        ]
        paper = predictor.generate_mock_paper(sample_questions, target_marks=8, target_questions=3)
        self.assertEqual(paper.total_marks, 8)
        self.assertEqual(len(paper.questions), 3)
        self.assertGreater(len(paper.sections), 0)
        self.assertIn("Section A", paper.sections[0]["section_name"])

    def test_minimum_study_set(self):
        """Verify greedy set-cover achieves 100% topic coverage with minimal questions."""
        ranked_questions = [
            {"id": 1, "topic_name": "Calculus", "marks": 5, "composite_score": 0.95},
            {"id": 2, "topic_name": "Calculus", "marks": 3, "composite_score": 0.90},
            {"id": 3, "topic_name": "Optics", "marks": 4, "composite_score": 0.85},
            {"id": 4, "topic_name": "Thermodynamics", "marks": 6, "composite_score": 0.80},
        ]
        study_set = predictor.compute_minimum_study_set(ranked_questions)
        self.assertEqual(study_set.target_topics_covered, 3)
        self.assertEqual(study_set.total_questions, 3)  # 1 from each topic
        self.assertEqual(study_set.coverage_percent, 100.0)
        self.assertIn("Calculus", study_set.covered_topics)
        self.assertIn("Optics", study_set.covered_topics)
        self.assertIn("Thermodynamics", study_set.covered_topics)


if __name__ == "__main__":
    unittest.main()
