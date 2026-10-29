"""
Unit Tests for Phase 6: Study Scheduler (engine/study_scheduler.py)
Tests SM-2 repetition logic, time estimation, day-by-day scheduling,
and progress persistence.
"""

from datetime import date, timedelta
import os
import tempfile
import unittest

from engine import db, study_scheduler


class TestStudyScheduler(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_study.db")
        db.init_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sm2_successful_progression(self):
        """Verify successful ratings increase repetition count, interval, and EF."""
        base = date(2026, 1, 1)
        # First review with rating 5 (perfect)
        r1 = study_scheduler.compute_sm2_update(rating=5, current_ef=2.5, current_interval=1, repetition_count=0, base_date=base)
        self.assertEqual(r1.repetition_count, 1)
        self.assertEqual(r1.interval_days, 1)
        self.assertEqual(r1.status, "confident")
        self.assertGreaterEqual(r1.easiness_factor, 2.5)

        # Second review with rating 5
        r2 = study_scheduler.compute_sm2_update(rating=5, current_ef=r1.easiness_factor, current_interval=r1.interval_days, repetition_count=r1.repetition_count, base_date=base)
        self.assertEqual(r2.repetition_count, 2)
        self.assertEqual(r2.interval_days, 3)

        # Third review with rating 5 -> should reach mastered
        r3 = study_scheduler.compute_sm2_update(rating=5, current_ef=r2.easiness_factor, current_interval=r2.interval_days, repetition_count=r2.repetition_count, base_date=base)
        self.assertEqual(r3.repetition_count, 3)
        self.assertGreater(r3.interval_days, 3)
        self.assertEqual(r3.status, "mastered")

    def test_sm2_failure_reset(self):
        """Verify poor recall (rating < 3) resets repetitions to 0 and marks as needs_review."""
        r_fail = study_scheduler.compute_sm2_update(
            rating=2,
            current_ef=2.4,
            current_interval=10,
            repetition_count=4,
        )
        self.assertEqual(r_fail.repetition_count, 0)
        self.assertEqual(r_fail.interval_days, 1)
        self.assertEqual(r_fail.status, "needs_review")
        self.assertLess(r_fail.easiness_factor, 2.4)

    def test_question_study_time_estimation(self):
        """Verify time budget adapts to question type, marks, and difficulty tier."""
        mcq_easy = {
            "question_type": "mcq",
            "marks": 1,
            "difficulty_tier": "Easy",
        }
        long_hard = {
            "question_type": "long_answer",
            "marks": 10,
            "difficulty_tier": "Hard",
        }

        t_mcq = study_scheduler.estimate_question_study_time_minutes(mcq_easy)
        t_long = study_scheduler.estimate_question_study_time_minutes(long_hard)

        self.assertLess(t_mcq, 10)
        self.assertGreater(t_long, 20)

    def test_generate_study_schedule_distribution(self):
        """Verify schedule allocates tasks across multiple days and schedules reviews."""
        exam_date = (date.today() + timedelta(days=7)).isoformat()
        sample_questions = [
            {"id": 1, "question_number": "1", "question_text": "Q1", "question_type": "mcq", "marks": 1, "topic_name": "Calculus", "composite_score": 0.9},
            {"id": 2, "question_number": "2", "question_text": "Q2", "question_type": "short_answer", "marks": 2, "topic_name": "Algebra", "composite_score": 0.8},
            {"id": 3, "question_number": "3", "question_text": "Q3", "question_type": "long_answer", "marks": 5, "topic_name": "Geometry", "composite_score": 0.7},
        ]

        plan = study_scheduler.generate_study_schedule(
            questions=sample_questions,
            target_exam_date=exam_date,
            daily_study_hours=2.0,
        )

        self.assertEqual(plan.days_remaining, 7)
        self.assertEqual(plan.total_questions, 3)
        self.assertEqual(len(plan.daily_schedule), 7)
        self.assertGreater(plan.total_study_minutes_estimated, 0)

        # Check that Day 1 has learning tasks
        day1 = plan.daily_schedule[0]
        self.assertGreater(len(day1.tasks), 0)
        self.assertTrue(any(t.task_type == "learn" for t in day1.tasks))

    def test_record_question_review_persistence(self):
        """Verify recording a review persists the updated SM-2 status to database."""
        doc_id = db.insert_document("Exam.pdf", "/tmp/ex.pdf", "hash1", "pdf", 100, year=2024, db_path=self.db_path)
        qid = db.insert_question(
            doc_id=doc_id,
            page_number=1,
            question_number="1",
            question_text="Sample Q",
            question_type="short_answer",
            marks=2,
            quality_score=90,
            db_path=self.db_path,
        )

        # Review question with rating 5 (confident)
        res = study_scheduler.record_question_review(qid, rating=5, db_path=self.db_path)
        self.assertEqual(res["status"], "confident")
        self.assertEqual(res["repetition_count"], 1)

        # Check that DB row was updated
        db_prog = db.get_study_progress_for_question(qid, db_path=self.db_path)
        self.assertIsNotNone(db_prog)
        self.assertEqual(db_prog["status"], "confident")
        self.assertEqual(db_prog["repetition_count"], 1)

        stats = db.get_study_stats(db_path=self.db_path)
        self.assertEqual(stats["confident"], 1)


if __name__ == "__main__":
    unittest.main()
