"""
Unit Tests for Ingestion Pipeline (engine/ingest.py)
"""

import os
import tempfile
import unittest
import zipfile
from engine import db, ingest

try:
    import pymupdf
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


class TestIngestionPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_engine.db")
        db.init_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compute_file_hash(self):
        """Verify deterministic SHA-256 computation."""
        test_file = os.path.join(self.temp_dir.name, "test_sample.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Deterministic content for hash test.")

        h1 = ingest.compute_file_hash(test_file)
        h2 = ingest.compute_file_hash(test_file)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)

    def test_detect_year_from_name(self):
        """Verify 4-digit exam year auto-detection."""
        self.assertEqual(ingest.detect_year_from_name("Math_2022_Final.pdf"), 2022)
        self.assertEqual(ingest.detect_year_from_name("Exam_Paper_1998.docx"), 1998)
        self.assertEqual(ingest.detect_year_from_name("NoYearPaper.pdf"), None)
        self.assertEqual(ingest.detect_year_from_name("Physics_2021_retake_2023.txt"), 2023)

    def test_extract_text_from_txt(self):
        """Verify TXT ingestion with page breaks."""
        txt_path = os.path.join(self.temp_dir.name, "sample.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Section 1: General Knowledge\x0cSection 2: Quantitative Aptitude")

        pages = ingest.extract_text_from_txt(txt_path)
        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0].page_number, 1)
        self.assertIn("Section 1", pages[0].text)
        self.assertEqual(pages[1].page_number, 2)
        self.assertIn("Section 2", pages[1].text)

    def test_extract_text_from_docx_xml(self):
        """Verify DOCX extraction using XML container."""
        docx_path = os.path.join(self.temp_dir.name, "test.docx")
        # Build minimal valid DOCX zip structure
        xml_content = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body>"
            "<w:p><w:r><w:t>Question 1: What is kinetic energy?</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>Question 2: State Newton's third law.</w:t></w:r></w:p>"
            "</w:body>"
            "</w:document>"
        )
        with zipfile.ZipFile(docx_path, "w") as z:
            z.writestr("word/document.xml", xml_content)

        pages = ingest.extract_text_from_docx(docx_path)
        self.assertEqual(len(pages), 1)
        self.assertIn("kinetic energy", pages[0].text)
        self.assertIn("Newton's third law", pages[0].text)

    def test_extract_text_from_pdf(self):
        """Verify PDF text extraction using PyMuPDF / pdfplumber."""
        if not HAS_PYMUPDF:
            self.skipTest("PyMuPDF not available")

        pdf_path = os.path.join(self.temp_dir.name, "sample.pdf")
        doc = pymupdf.open()
        p1 = doc.new_page()
        p1.insert_text((50, 72), "Mathematics Exam 2024 - Section A: Multiple Choice Questions")
        p2 = doc.new_page()
        p2.insert_text((50, 72), "Mathematics Exam 2024 - Section B: Long Answer Questions")
        doc.save(pdf_path)
        doc.close()

        pages = ingest.extract_text_from_pdf(pdf_path)
        self.assertEqual(len(pages), 2)
        self.assertIn("Section A", pages[0].text)
        self.assertIn("Section B", pages[1].text)

    def test_incremental_ingestion_and_deduplication(self):
        """Verify incremental processing skips existing files unless forced."""
        txt_path = os.path.join(self.temp_dir.name, "Biology_2023.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Q1: Explain cellular respiration in mitochondria.")

        # 1. First Ingestion
        res1 = ingest.ingest_file(txt_path, db_path=self.db_path)
        self.assertFalse(res1.skipped)
        self.assertEqual(res1.status, "ingested")
        self.assertEqual(res1.total_pages, 1)

        doc = db.get_document_by_id(res1.document_id, db_path=self.db_path)
        self.assertEqual(doc["status"], "ingested")
        self.assertEqual(doc["year"], 2023)

        # 2. Second Ingestion without force -> Should skip
        res2 = ingest.ingest_file(txt_path, force_reingest=False, db_path=self.db_path)
        self.assertTrue(res2.skipped)
        self.assertEqual(res2.document_id, res1.document_id)

        # 3. Third Ingestion with force_reingest=True -> Should re-process
        res3 = ingest.ingest_file(txt_path, force_reingest=True, db_path=self.db_path)
        self.assertFalse(res3.skipped)
        self.assertEqual(res3.document_id, res1.document_id)

    def test_unsupported_extension(self):
        """Verify unsupported extensions raise an error."""
        invalid_path = os.path.join(self.temp_dir.name, "script.exe")
        with open(invalid_path, "w") as f:
            f.write("fake binary")

        with self.assertRaises(ValueError):
            ingest.ingest_file(invalid_path, db_path=self.db_path)


if __name__ == "__main__":
    unittest.main()
