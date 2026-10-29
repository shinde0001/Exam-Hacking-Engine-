"""
Exam-Hacking Engine — Vector Store & Semantic Search
Encodes question texts into vector space and performs high-speed similarity search using FAISS.
Features:
- Dual Embedding Backends:
  * Neural embeddings via sentence-transformers (all-MiniLM-L6-v2) when available
  * Dense semantic TF-IDF + TruncatedSVD L2-normalized vectors (Scikit-Learn) fallback
- FAISS Index:
  * IndexFlatIP for exact cosine similarity
  * IVFFlat support for 10k+ scale
- Semantic search across the question bank
- Complete corpus deduplication scanner
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import config
from engine import db, similarity

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class VectorStore:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.SENTENCE_TRANSFORMER_MODEL
        self.st_model = None
        self.tfidf_vectorizer = None
        self.svd_model = None
        self.index = None
        self.dimension = 0
        self.question_ids: List[int] = []

        self._init_encoder()

    def _init_encoder(self):
        """Initialize sentence-transformer model if available, else TF-IDF/SVD."""
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.st_model = SentenceTransformer(self.model_name)
                # Probe embedding dimension
                probe = self.st_model.encode(["probe"], show_progress_bar=False)
                self.dimension = probe.shape[1]
                return
            except Exception as e:
                print(f"[VectorStore] SentenceTransformer init error: {e}. Falling back to TF-IDF/SVD.")

        # Default dimension for dense TF-IDF/SVD fallback
        self.dimension = 64

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode list of strings into L2-normalized float32 numpy array.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Pre-process text using similarity normalization
        cleaned_texts = [similarity.normalize_text_for_comparison(t) for t in texts]

        # 1. Neural Encoder
        if self.st_model is not None:
            embeddings = self.st_model.encode(
                cleaned_texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.astype(np.float32)

        # 2. Scikit-Learn TF-IDF + SVD Fallback
        if HAS_SKLEARN:
            if self.tfidf_vectorizer is None:
                # Fit new vectorizer
                self.tfidf_vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=2000,
                    sublinear_tf=True,
                    stop_words="english",
                )
                tfidf_mat = self.tfidf_vectorizer.fit_transform(cleaned_texts)
            else:
                try:
                    tfidf_mat = self.tfidf_vectorizer.transform(cleaned_texts)
                except Exception:
                    self.tfidf_vectorizer = TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=2000,
                        sublinear_tf=True,
                        stop_words="english",
                    )
                    tfidf_mat = self.tfidf_vectorizer.fit_transform(cleaned_texts)

            n_samples, n_features = tfidf_mat.shape
            n_components = min(self.dimension, n_features - 1) if n_features > 1 else 1

            if self.svd_model is None or self.svd_model.n_components != n_components:
                self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
                dense = self.svd_model.fit_transform(tfidf_mat)
            else:
                dense = self.svd_model.transform(tfidf_mat)

            # Pad with zeros if n_components < self.dimension
            if dense.shape[1] < self.dimension:
                pad_width = self.dimension - dense.shape[1]
                dense = np.pad(dense, ((0, 0), (0, pad_width)), mode="constant")

            # L2 normalize so dot product equals cosine similarity
            normed = normalize(dense, norm="l2", axis=1)
            return normed.astype(np.float32)

        # 3. Simple character n-gram hashing fallback
        vectors = []
        for t in cleaned_texts:
            v = np.zeros(self.dimension, dtype=np.float32)
            words = t.split()
            for w in words:
                idx = hash(w) % self.dimension
                v[idx] += 1.0
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            vectors.append(v)
        return np.vstack(vectors)

    def build_index(self, questions: List[Dict[str, Any]]):
        """
        Build FAISS index from a list of question records.
        """
        if not questions:
            self.index = None
            self.question_ids = []
            return

        # Reset vectorizer so it learns the new corpus vocabulary
        if HAS_SKLEARN and self.st_model is None:
            self.tfidf_vectorizer = None
            self.svd_model = None

        texts = [q["question_text"] for q in questions]
        embeddings = self.encode(texts)

        self.question_ids = [q["id"] for q in questions]

        # Initialize FAISS Index
        if HAS_FAISS:
            num_vectors = len(questions)
            if num_vectors >= 10000:
                # Use IVF for large-scale datasets (>10k)
                nlist = min(100, int(np.sqrt(num_vectors)))
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                self.index.train(embeddings)
            else:
                # Use exact FlatIP for maximum precision
                self.index = faiss.IndexFlatIP(self.dimension)

            self.index.add(embeddings)
            del embeddings
        else:
            # Keep embeddings in memory for numpy search
            self.index = embeddings

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """
        Search the index for the most similar questions.
        Returns list of (question_id, similarity_score).
        """
        if not self.question_ids or (self.index is None):
            return []

        query_vec = self.encode([query])

        if HAS_FAISS and isinstance(self.index, faiss.Index):
            k = min(top_k, len(self.question_ids))
            scores, indices = self.index.search(query_vec, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.question_ids):
                    if score >= min_score:
                        results.append((self.question_ids[idx], float(score)))
            return results

        # Fallback numpy matrix inner product
        if isinstance(self.index, np.ndarray):
            sims = np.dot(self.index, query_vec.T).flatten()
            top_indices = np.argsort(-sims)[:top_k]
            results = []
            for idx in top_indices:
                score = float(sims[idx])
                if score >= min_score:
                    results.append((self.question_ids[idx], score))
            return results

        return []

    def save_index(self, base_path: Optional[str] = None):
        """Save FAISS index and mapping to disk."""
        target_dir = base_path or config.DATABASE_DIR
        os.makedirs(target_dir, exist_ok=True)

        mapping_file = os.path.join(target_dir, "vector_mapping.json")
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump({"question_ids": self.question_ids, "dimension": self.dimension}, f)

        if HAS_FAISS and self.index is not None and isinstance(self.index, faiss.Index):
            faiss_file = os.path.join(target_dir, "vector_index.faiss")
            faiss.write_index(self.index, faiss_file)

    def load_index(self, base_path: Optional[str] = None) -> bool:
        """Load FAISS index and mapping from disk."""
        target_dir = base_path or config.DATABASE_DIR
        mapping_file = os.path.join(target_dir, "vector_mapping.json")
        faiss_file = os.path.join(target_dir, "vector_index.faiss")

        if not os.path.exists(mapping_file):
            return False

        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.question_ids = data["question_ids"]
                self.dimension = data.get("dimension", 64)

            if HAS_FAISS and os.path.exists(faiss_file):
                self.index = faiss.read_index(faiss_file)
                return True
        except Exception as e:
            print(f"[VectorStore] Error loading index: {e}")

        return False


# Global vector store instance
_GLOBAL_STORE: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Return singleton vector store instance."""
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = VectorStore()
    return _GLOBAL_STORE


def reset_vector_store():
    """Reset the singleton vector store instance."""
    global _GLOBAL_STORE
    _GLOBAL_STORE = None


def build_and_save_vector_index(db_path: Optional[str] = None) -> int:
    """
    Extracts all valid questions from database, builds the vector store,
    and persists index to disk. Returns number of indexed questions.
    """
    questions = db.list_questions(valid_only=True, db_path=db_path)
    if not questions:
        return 0

    store = get_vector_store()
    store.build_index(questions)
    store.save_index()
    return len(questions)


def find_and_record_duplicates(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Full-corpus deduplication & conflict scanner:
    1. Loads all valid questions.
    2. Builds/updates vector index.
    3. Finds top-K nearest neighbors per question.
    4. Evaluates 3-layer similarity (exact, near-exact, semantic) and checks answer conflicts.
    5. Stores discovered duplicate edges in SQLite.
    Returns list of discovered duplicate edges.
    """
    questions = db.list_questions(valid_only=True, db_path=db_path)
    if len(questions) < 2:
        return []

    store = get_vector_store()
    store.build_index(questions)

    q_lookup = {q["id"]: q for q in questions}
    detected_edges = []
    seen_pairs = set()

    # Clear old edges before full re-scan
    db.delete_similarity_edges(db_path=db_path)

    top_k = min(config.SIMILARITY_TOP_K, len(questions))

    for q in questions:
        q_id = q["id"]
        # Search for candidates
        candidates = store.search(q["question_text"], top_k=top_k)

        for candidate_id, semantic_sim in candidates:
            if candidate_id == q_id:
                continue

            pair_key = tuple(sorted([q_id, candidate_id]))
            if pair_key in seen_pairs:
                continue

            target_q = q_lookup[candidate_id]

            # 3-Layer comparison
            sim_result = similarity.compare_question_pair(
                q1_text=q["question_text"],
                q2_text=target_q["question_text"],
                ans1=q["answer"],
                ans2=target_q["answer"],
                semantic_score=semantic_sim,
            )

            if sim_result.is_duplicate:
                seen_pairs.add(pair_key)
                edge_id = db.insert_similarity_edge(
                    source_id=q_id,
                    target_id=candidate_id,
                    layer=sim_result.layer,
                    score=sim_result.score,
                    has_conflict=sim_result.has_conflict,
                    conflict_note=sim_result.conflict_note,
                    db_path=db_path,
                )

                detected_edges.append({
                    "id": edge_id,
                    "source_id": q_id,
                    "target_id": candidate_id,
                    "source_text": q["question_text"],
                    "target_text": target_q["question_text"],
                    "layer": sim_result.layer,
                    "score": sim_result.score,
                    "has_conflict": sim_result.has_conflict,
                    "conflict_note": sim_result.conflict_note,
                })

    return detected_edges


def semantic_search_questions(
    query: str,
    top_k: int = 10,
    min_score: float = 0.5,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search the question bank with a natural language query.
    Returns hydrated question dictionaries sorted by similarity score.
    """
    store = get_vector_store()
    if store.index is None:
        # Build index on the fly if not yet built
        build_and_save_vector_index(db_path=db_path)

    matches = store.search(query, top_k=top_k, min_score=min_score)
    if not matches:
        return []

    results = []
    for q_id, score in matches:
        q_record = db.get_question_by_id(q_id, db_path=db_path)
        if q_record:
            doc_record = db.get_document_by_id(q_record["document_id"], db_path=db_path)
            results.append({
                **q_record,
                "similarity_score": round(score, 4),
                "document_filename": doc_record["filename"] if doc_record else "Unknown",
                "document_year": doc_record["year"] if doc_record else None,
            })

    return results
