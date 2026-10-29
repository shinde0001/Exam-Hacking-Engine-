"""
Unit Tests for Vector Store & Semantic Search (engine/vectorstore.py)
"""

import os
import tempfile
import unittest
import numpy as np
from engine import db, vectorstore


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_engine.db")
        db.init_db(self.db_path)
        self.store = vectorstore.VectorStore()

    def tearDown(self):
        vectorstore.reset_vector_store()
        self.temp_dir.cleanup()

    def test_encode_and_normalization(self):
        """Verify embeddings are L2 normalized (norm ~= 1.0)."""
        texts = [
            "Calculate kinetic energy of moving particle",
            "State Heisenberg uncertainty principle in quantum physics",
        ]
        embeddings = self.store.encode(texts)
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings.dtype, np.float32)

        # Check unit norm
        for v in embeddings:
            norm = np.linalg.norm(v)
            self.assertAlmostEqual(norm, 1.0, places=4)

    def test_build_index_and_search(self):
        """Verify FAISS index construction and nearest neighbor search."""
        questions = [
            {"id": 101, "question_text": "State Newton's first law of motion and inertia"},
            {"id": 102, "question_text": "Calculate velocity and acceleration of falling stone"},
            {"id": 103, "question_text": "Explain photosynthesis and light reactions in chloroplast"},
        ]
        self.store.build_index(questions)

        # Query closely related to question 101
        results = self.store.search("Newton inertia law", top_k=2)
        self.assertGreater(len(results), 0)
        top_id, top_score = results[0]
        self.assertEqual(top_id, 101)
        self.assertGreater(top_score, 0.4)

    def test_save_and_load_index(self):
        """Verify saving and restoring FAISS index and mapping."""
        questions = [
            {"id": 201, "question_text": "Define electric flux and Gauss theorem"},
            {"id": 202, "question_text": "Derive expression for capacitance of parallel plate"},
        ]
        self.store.build_index(questions)
        self.store.save_index(base_path=self.temp_dir.name)

        new_store = vectorstore.VectorStore()
        loaded = new_store.load_index(base_path=self.temp_dir.name)
        self.assertTrue(loaded)
        self.assertEqual(new_store.question_ids, [201, 202])

    def test_find_and_record_duplicates_pipeline(self):
        """Verify full duplicate scanning and SQLite edge insertion."""
        # Insert test document
        doc1 = db.insert_document("Book_A.pdf", "/tmp/A.pdf", "hashA", "pdf", 100, year=2021, db_path=self.db_path)
        doc2 = db.insert_document("Book_B.pdf", "/tmp/B.pdf", "hashB", "pdf", 100, year=2022, db_path=self.db_path)

        # Insert questions (including an exact duplicate with conflicting answers!)
        q1_id = db.insert_question(
            doc_id=doc1,
            page_number=1,
            question_number="1",
            question_text="What is the unit of electric resistance?",
            question_type="mcq",
            answer="Ohm",
            quality_score=90,
            db_path=self.db_path,
        )
        q2_id = db.insert_question(
            doc_id=doc2,
            page_number=3,
            question_number="5",
            question_text="What is the unit of electric resistance?",  # Exact repeat
            question_type="mcq",
            answer="Volt",  # Conflicting answer!
            quality_score=90,
            db_path=self.db_path,
        )

        edges = vectorstore.find_and_record_duplicates(db_path=self.db_path)
        self.assertEqual(len(edges), 1)

        edge = edges[0]
        self.assertEqual(edge["layer"], "exact")
        self.assertEqual(edge["score"], 1.0)
        self.assertTrue(edge["has_conflict"])
        self.assertIn("Ohm", edge["conflict_note"])

        # Check database records
        db_edges = db.list_similarity_edges(db_path=self.db_path)
        self.assertEqual(len(db_edges), 1)
        self.assertEqual(db_edges[0]["has_answer_conflict"], 1)


if __name__ == "__main__":
    unittest.main()
