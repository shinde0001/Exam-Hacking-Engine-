"""
Unit Tests for Flask Web Application (app.py)
Tests all UI pages, API endpoints, extraction and duplicate actions,
and artifact export endpoints.
"""

import io
import os
import tempfile
import unittest
from unittest.mock import patch

import config
from app import app
from engine import db


class TestWebApp(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_engine.db")
        db.init_db(self.db_path)

        # Configure app for testing
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()

        # Patch config DATABASE_PATH, UPLOAD_DIR, and EXPORT_DIR
        self.export_dir = os.path.join(self.temp_dir.name, "exports")
        os.makedirs(self.export_dir, exist_ok=True)

        self.patch_db = patch("config.DATABASE_PATH", self.db_path)
        self.patch_upload = patch("config.UPLOAD_DIR", self.temp_dir.name)
        self.patch_export = patch("config.EXPORT_DIR", self.export_dir)
        self.patch_db.start()
        self.patch_upload.start()
        self.patch_export.start()

    def tearDown(self):
        from engine import vectorstore
        vectorstore.reset_vector_store()
        self.patch_export.stop()
        self.patch_upload.stop()
        self.patch_db.stop()
        self.temp_dir.cleanup()

    def test_dashboard_route(self):
        """Verify dashboard loads with HTTP 200."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Ingestion Dashboard", response.data)

    def test_upload_page_route(self):
        """Verify upload form loads with HTTP 200."""
        response = self.client.get("/upload")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Upload Exam Documents", response.data)

    def test_api_health(self):
        """Verify health check JSON endpoint."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertEqual(json_data["status"], "healthy")

    def test_upload_file_flow(self):
        """Verify uploading a document through the web form."""
        data = {
            "files": (io.BytesIO(b"Section A: Questions\n1. What is gravity?"), "Physics_2021.txt"),
            "year": "2021",
            "subject": "Physics",
        }
        response = self.client.post("/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Physics_2021.txt", response.data)

        # Verify through API
        api_res = self.client.get("/api/documents")
        self.assertEqual(api_res.status_code, 200)
        docs = api_res.get_json()["documents"]
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["filename"], "Physics_2021.txt")

    def test_document_views_and_extraction_flow(self):
        """Verify document detail, question extraction, and questions view."""
        sample_exam_text = (
            "Question 1. What is the SI unit of electrical resistance? [1 Mark]\n"
            "(A) Volt\n(B) Ampere\n(C) Ohm\n(D) Henry\nAns: C\n\n"
            "Question 2. Explain electromagnetic induction and derive Faraday's law. [5 Marks]\n"
        )
        doc_path = os.path.join(self.temp_dir.name, "Exam_2024.txt")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(sample_exam_text)

        doc_id = db.insert_document("Exam_2024.txt", doc_path, "hash2024", "txt", len(sample_exam_text), year=2024, db_path=self.db_path)
        db.insert_document_page(doc_id, 1, sample_exam_text, len(sample_exam_text), "text_read", False, db_path=self.db_path)

        # 1. View Document
        res_view = self.client.get(f"/documents/{doc_id}")
        self.assertEqual(res_view.status_code, 200)
        self.assertIn(b"Exam_2024.txt", res_view.data)

        # 2. Extract Questions
        res_extract = self.client.post(f"/documents/{doc_id}/extract", follow_redirects=True)
        self.assertEqual(res_extract.status_code, 200)

        # 3. View Extracted Questions
        res_qs = self.client.get(f"/documents/{doc_id}/questions")
        self.assertEqual(res_qs.status_code, 200)
        self.assertIn(b"electrical resistance", res_qs.data)

        # 4. API Document Questions
        res_api_qs = self.client.get(f"/api/documents/{doc_id}/questions")
        self.assertEqual(res_api_qs.status_code, 200)
        self.assertEqual(res_api_qs.get_json()["count"], 2)

    def test_phase3_semantic_search_and_duplicates(self):
        """Verify Phase 3 search and deduplication pages and APIs."""
        # Insert a sample question
        doc_id = db.insert_document("Doc.pdf", "/tmp/d.pdf", "hd", "pdf", 100, year=2023, db_path=self.db_path)
        db.insert_question(doc_id, 1, "1", "State the second law of thermodynamics.", "short_answer", marks=2, quality_score=90, db_path=self.db_path)

        # Search GET
        res_search = self.client.get("/search?q=thermodynamics")
        self.assertEqual(res_search.status_code, 200)

        # API Search
        res_api_search = self.client.get("/api/search?q=thermodynamics")
        self.assertEqual(res_api_search.status_code, 200)

        # Duplicates page & scan
        res_dup = self.client.get("/duplicates")
        self.assertEqual(res_dup.status_code, 200)

        res_scan = self.client.post("/duplicates/scan", follow_redirects=True)
        self.assertEqual(res_scan.status_code, 200)

        # API Duplicates
        res_api_dup = self.client.get("/api/duplicates")
        self.assertEqual(res_api_dup.status_code, 200)

    def test_phase4_patterns_and_syllabus(self):
        """Verify Phase 4 pattern analysis and syllabus mapper pages and APIs."""
        # Patterns page (works even on empty DB thanks to bug fix)
        res_pat = self.client.get("/patterns")
        self.assertEqual(res_pat.status_code, 200)

        res_api_pat = self.client.get("/api/patterns")
        self.assertEqual(res_api_pat.status_code, 200)

        # Syllabus page GET and POST
        res_syl = self.client.get("/syllabus")
        self.assertEqual(res_syl.status_code, 200)

        syl_text = "Chapter 1: Heat and Thermodynamics\n- Second Law\n- Entropy"
        res_syl_post = self.client.post("/syllabus", data={"syllabus_text": syl_text})
        self.assertEqual(res_syl_post.status_code, 200)

        res_api_syl = self.client.post("/api/syllabus", json={"syllabus_text": syl_text})
        self.assertEqual(res_api_syl.status_code, 200)

    def test_phase5_predictions_and_mock_paper(self):
        """Verify Phase 5 prediction report and mock paper generation."""
        doc_id = db.insert_document("Exam.pdf", "/tmp/e.pdf", "he", "pdf", 100, year=2024, db_path=self.db_path)
        db.insert_question(doc_id, 1, "1", "What is Newton's second law?", "short_answer", marks=2, quality_score=90, db_path=self.db_path)

        res_pred = self.client.get("/predictions")
        self.assertEqual(res_pred.status_code, 200)

        res_api_pred = self.client.get("/api/predictions")
        self.assertEqual(res_api_pred.status_code, 200)

        # Mock paper page GET and POST
        res_mock = self.client.get("/mock-paper")
        self.assertEqual(res_mock.status_code, 200)

        res_mock_post = self.client.post("/mock-paper", data={"target_marks": "20", "target_questions": "5"})
        self.assertEqual(res_mock_post.status_code, 200)

        res_api_mock = self.client.post("/api/predictions/mock-paper", json={"target_marks": 20, "target_questions": 5})
        self.assertEqual(res_api_mock.status_code, 200)

        res_api_set = self.client.get("/api/predictions/minimum-study-set")
        self.assertEqual(res_api_set.status_code, 200)

    def test_phase6_study_plan_and_reviews(self):
        """Verify Phase 6 study planner page and spaced repetition review updates."""
        doc_id = db.insert_document("Exam.pdf", "/tmp/e.pdf", "he", "pdf", 100, year=2024, db_path=self.db_path)
        q_id = db.insert_question(doc_id, 1, "1", "What is kinetic energy?", "short_answer", marks=2, quality_score=90, db_path=self.db_path)

        res_plan = self.client.get("/study-plan")
        self.assertEqual(res_plan.status_code, 200)

        res_plan_post = self.client.post("/study-plan", data={"daily_study_hours": "3.0"})
        self.assertEqual(res_plan_post.status_code, 200)

        res_api_plan = self.client.get("/api/study-plan")
        self.assertEqual(res_api_plan.status_code, 200)

        # Record review (SM-2 update)
        res_rev = self.client.post("/api/study-plan/record-review", json={"question_id": q_id, "rating": 5})
        self.assertEqual(res_rev.status_code, 200)
        self.assertEqual(res_rev.get_json()["status"], "confident")

    def test_phase7_export_endpoints(self):
        """Verify all export endpoints return valid file attachments."""
        doc_id = db.insert_document("Exam.pdf", "/tmp/e.pdf", "he", "pdf", 100, year=2024, db_path=self.db_path)
        db.insert_question(doc_id, 1, "1", "Define momentum and energy.", "short_answer", marks=2, quality_score=90, db_path=self.db_path)

        # CSV questions export
        res_csv = self.client.get("/export/questions/csv")
        self.assertEqual(res_csv.status_code, 200)
        self.assertEqual(res_csv.mimetype, "text/csv")
        self.assertIn(b"Question ID", res_csv.data)

        # Anki export
        res_anki = self.client.get("/export/questions/anki")
        self.assertEqual(res_anki.status_code, 200)
        self.assertIn(b"ExamEngine", res_anki.data)

        # Mock Paper PDF export
        res_pdf = self.client.get("/export/mock-paper/pdf?target_marks=20&target_questions=5")
        self.assertEqual(res_pdf.status_code, 200)
        self.assertEqual(res_pdf.mimetype, "application/pdf")
        self.assertTrue(res_pdf.data.startswith(b"%PDF-"))
        res_pdf.close()

        # Study Plan CSV export
        res_study_csv = self.client.get("/export/study-plan/csv")
        self.assertEqual(res_study_csv.status_code, 200)
        self.assertEqual(res_study_csv.mimetype, "text/csv")

    def test_document_reingest_and_delete(self):
        """Verify reingestion and deletion workflows."""
        filepath = os.path.join(self.temp_dir.name, "SampleDoc.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("Q1: Explain photosynthesis.")

        doc_id = db.insert_document("SampleDoc.txt", filepath, "samplehash", "txt", 30, year=2023, db_path=self.db_path)

        # Reingest
        res_reingest = self.client.post(f"/documents/{doc_id}/reingest", follow_redirects=True)
        self.assertEqual(res_reingest.status_code, 200)

        # Delete
        res_delete = self.client.post(f"/documents/{doc_id}/delete", follow_redirects=True)
        self.assertEqual(res_delete.status_code, 200)
        self.assertIsNone(db.get_document_by_id(doc_id, db_path=self.db_path))


if __name__ == "__main__":
    unittest.main()
