"""
Unit Tests for Database Management Layer (engine/db.py)
"""

import os
import tempfile
import unittest
from engine import db


class TestDatabaseLayer(unittest.TestCase):
    def setUp(self):
        # Create a temporary database file for isolation
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_engine.db")
        db.init_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_schema_initialization(self):
        """Verify tables and indices are created."""
        with db.get_db_connection(self.db_path) as conn:
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            self.assertIn("documents", tables)
            self.assertIn("document_pages", tables)

    def test_insert_and_get_document(self):
        """Verify inserting document and retrieving by ID and hash."""
        doc_id = db.insert_document(
            filename="Math_2023.pdf",
            filepath="/tmp/Math_2023.pdf",
            file_hash="abc123hash",
            file_type="pdf",
            file_size=1024,
            year=2023,
            subject="Mathematics",
            db_path=self.db_path,
        )
        self.assertGreater(doc_id, 0)

        # Get by ID
        doc = db.get_document_by_id(doc_id, db_path=self.db_path)
        self.assertIsNotNone(doc)
        self.assertEqual(doc["filename"], "Math_2023.pdf")
        self.assertEqual(doc["year"], 2023)
        self.assertEqual(doc["status"], "pending")

        # Get by hash
        doc_by_hash = db.get_document_by_hash("abc123hash", db_path=self.db_path)
        self.assertIsNotNone(doc_by_hash)
        self.assertEqual(doc_by_hash["id"], doc_id)

    def test_update_status(self):
        """Verify updating document status, page count, and errors."""
        doc_id = db.insert_document(
            filename="Physics.pdf",
            filepath="/tmp/Physics.pdf",
            file_hash="phyhash",
            file_type="pdf",
            file_size=2048,
            db_path=self.db_path,
        )

        db.update_document_status(doc_id, "ingested", total_pages=5, db_path=self.db_path)
        doc = db.get_document_by_id(doc_id, db_path=self.db_path)
        self.assertEqual(doc["status"], "ingested")
        self.assertEqual(doc["total_pages"], 5)

        # Update to failed with error message
        db.update_document_status(
            doc_id, "failed", error_message="Corrupted file", db_path=self.db_path
        )
        doc = db.get_document_by_id(doc_id, db_path=self.db_path)
        self.assertEqual(doc["status"], "failed")
        self.assertEqual(doc["error_message"], "Corrupted file")

    def test_insert_and_get_document_pages(self):
        """Verify inserting pages and fetching in sequential order."""
        doc_id = db.insert_document(
            filename="Chemistry.txt",
            filepath="/tmp/Chemistry.txt",
            file_hash="chemhash",
            file_type="txt",
            file_size=500,
            db_path=self.db_path,
        )

        db.insert_document_page(
            doc_id=doc_id,
            page_number=1,
            raw_text="Page 1 text content",
            char_count=19,
            extraction_method="text_read",
            ocr_applied=False,
            db_path=self.db_path,
        )
        db.insert_document_page(
            doc_id=doc_id,
            page_number=2,
            raw_text="Page 2 text content",
            char_count=19,
            extraction_method="text_read",
            ocr_applied=True,
            db_path=self.db_path,
        )

        pages = db.get_document_pages(doc_id, db_path=self.db_path)
        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertEqual(pages[0]["ocr_applied"], 0)
        self.assertEqual(pages[1]["page_number"], 2)
        self.assertEqual(pages[1]["ocr_applied"], 1)

    def test_database_stats(self):
        """Verify aggregation stats."""
        doc1 = db.insert_document("A.pdf", "/tmp/A.pdf", "hashA", "pdf", 100, db_path=self.db_path)
        doc2 = db.insert_document("B.pdf", "/tmp/B.pdf", "hashB", "pdf", 200, db_path=self.db_path)

        db.update_document_status(doc1, "ingested", total_pages=2, db_path=self.db_path)
        db.update_document_status(doc2, "failed", error_message="Error", db_path=self.db_path)

        db.insert_document_page(doc1, 1, "Page 1", 6, "pdfplumber", False, db_path=self.db_path)
        db.insert_document_page(doc1, 2, "Page 2", 6, "ocr_tesseract", True, db_path=self.db_path)

        stats = db.get_database_stats(db_path=self.db_path)
        self.assertEqual(stats["total_documents"], 2)
        self.assertEqual(stats["ingested_documents"], 1)
        self.assertEqual(stats["failed_documents"], 1)
        self.assertEqual(stats["total_pages"], 2)
        self.assertEqual(stats["ocr_pages"], 1)

    def test_cascade_delete(self):
        """Verify deleting a document also deletes its associated pages."""
        doc_id = db.insert_document("Del.pdf", "/tmp/Del.pdf", "delhash", "pdf", 100, db_path=self.db_path)
        db.insert_document_page(doc_id, 1, "Content", 7, "pdfplumber", False, db_path=self.db_path)

        self.assertEqual(len(db.get_document_pages(doc_id, db_path=self.db_path)), 1)
        db.delete_document(doc_id, db_path=self.db_path)
        self.assertIsNone(db.get_document_by_id(doc_id, db_path=self.db_path))
        self.assertEqual(len(db.get_document_pages(doc_id, db_path=self.db_path)), 0)


if __name__ == "__main__":
    unittest.main()
