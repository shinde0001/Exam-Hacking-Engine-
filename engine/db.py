"""
Exam-Hacking Engine — Database Management
Handles SQLite database connection, initialization, migrations, and CRUD operations.
"""

import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
import config


def get_db_path(custom_path: Optional[str] = None) -> str:
    """Return configured database path or custom path if specified."""
    return custom_path or config.DATABASE_PATH


@contextmanager
def get_db_connection(db_path: Optional[str] = None):
    """
    Context manager providing a SQLite database connection with row factory
    and safety PRAGMAs (foreign keys and WAL mode).
    """
    path = get_db_path(db_path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[str] = None) -> None:
    """
    Initialize SQLite database tables from database/schema.sql.
    """
    schema_path = os.path.join(config.BASE_DIR, "database", "schema.sql")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found at: {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    with get_db_connection(db_path) as conn:
        conn.executescript(schema_sql)


def insert_document(
    filename: str,
    filepath: str,
    file_hash: str,
    file_type: str,
    file_size: int,
    year: Optional[int] = None,
    subject: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """
    Insert a new document record. Returns inserted document ID.
    """
    sql = """
        INSERT INTO documents (filename, filepath, file_hash, file_type, file_size, year, subject, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (filename, filepath, file_hash, file_type, file_size, year, subject))
        return cursor.lastrowid


def get_document_by_id(doc_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch document metadata by ID."""
    sql = "SELECT * FROM documents WHERE id = ?"
    with get_db_connection(db_path) as conn:
        row = conn.execute(sql, (doc_id,)).fetchone()
        return dict(row) if row else None


def get_document_by_hash(file_hash: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch document metadata by SHA-256 hash."""
    sql = "SELECT * FROM documents WHERE file_hash = ?"
    with get_db_connection(db_path) as conn:
        row = conn.execute(sql, (file_hash,)).fetchone()
        return dict(row) if row else None


def update_document_status(
    doc_id: int,
    status: str,
    total_pages: Optional[int] = None,
    error_message: Optional[str] = None,
    db_path: Optional[str] = None,
) -> bool:
    """
    Update document processing status and optional total pages or error message.
    """
    updates = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
    params: List[Any] = [status]

    if total_pages is not None:
        updates.append("total_pages = ?")
        params.append(total_pages)

    if error_message is not None:
        updates.append("error_message = ?")
        params.append(error_message)

    params.append(doc_id)
    sql = f"UPDATE documents SET {', '.join(updates)} WHERE id = ?"

    with get_db_connection(db_path) as conn:
        cursor = conn.execute(sql, params)
        return cursor.rowcount > 0


def insert_document_page(
    doc_id: int,
    page_number: int,
    raw_text: str,
    char_count: int,
    extraction_method: str,
    ocr_applied: bool = False,
    db_path: Optional[str] = None,
) -> int:
    """
    Insert an extracted page for a document. Replaces on conflict.
    """
    sql = """
        INSERT INTO document_pages (
            document_id, page_number, raw_text, char_count, extraction_method, ocr_applied
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(document_id, page_number) DO UPDATE SET
            raw_text = excluded.raw_text,
            char_count = excluded.char_count,
            extraction_method = excluded.extraction_method,
            ocr_applied = excluded.ocr_applied
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            sql,
            (doc_id, page_number, raw_text, char_count, extraction_method, 1 if ocr_applied else 0),
        )
        return cursor.lastrowid


def get_document_pages(doc_id: int, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all pages of a document ordered by page_number."""
    sql = "SELECT * FROM document_pages WHERE document_id = ? ORDER BY page_number ASC"
    with get_db_connection(db_path) as conn:
        rows = conn.execute(sql, (doc_id,)).fetchall()
        return [dict(r) for r in rows]


def list_documents(status: Optional[str] = None, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """List documents optionally filtered by processing status."""
    if status:
        sql = "SELECT * FROM documents WHERE status = ? ORDER BY created_at DESC"
        params = (status,)
    else:
        sql = "SELECT * FROM documents ORDER BY created_at DESC"
        params = ()

    with get_db_connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def delete_document(doc_id: int, db_path: Optional[str] = None) -> bool:
    """Delete document and its associated pages (via foreign key cascade)."""
    sql = "DELETE FROM documents WHERE id = ?"
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(sql, (doc_id,))
        return cursor.rowcount > 0


# ─── Questions CRUD ──────────────────────────────────────────────

def insert_question(
    doc_id: int,
    page_number: int,
    question_number: str,
    question_text: str,
    question_type: str,
    options_json: Optional[str] = None,
    answer: Optional[str] = None,
    marks: Optional[int] = None,
    quality_score: int = 0,
    is_valid: bool = True,
    validation_issues: Optional[str] = None,
    has_math: bool = False,
    normalized_text: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """
    Insert a structured question record. Returns inserted question ID.
    """
    sql = """
        INSERT INTO questions (
            document_id, page_number, question_number, question_text, question_type,
            options_json, answer, marks, quality_score, is_valid, validation_issues,
            has_math, normalized_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            sql,
            (
                doc_id,
                page_number,
                question_number,
                question_text,
                question_type,
                options_json,
                answer,
                marks,
                quality_score,
                1 if is_valid else 0,
                validation_issues,
                1 if has_math else 0,
                normalized_text or question_text,
            ),
        )
        return cursor.lastrowid


def get_question_by_id(question_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a single question by ID."""
    sql = "SELECT * FROM questions WHERE id = ?"
    with get_db_connection(db_path) as conn:
        row = conn.execute(sql, (question_id,)).fetchone()
        return dict(row) if row else None


def get_questions_by_document(
    doc_id: int,
    valid_only: bool = False,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve all questions for a given document ordered by id."""
    if valid_only:
        sql = "SELECT * FROM questions WHERE document_id = ? AND is_valid = 1 ORDER BY id ASC"
    else:
        sql = "SELECT * FROM questions WHERE document_id = ? ORDER BY id ASC"

    with get_db_connection(db_path) as conn:
        rows = conn.execute(sql, (doc_id,)).fetchall()
        return [dict(r) for r in rows]


def list_questions(
    question_type: Optional[str] = None,
    min_quality: Optional[int] = None,
    valid_only: bool = True,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List questions with optional filters on type, quality, and validity."""
    clauses = []
    params: List[Any] = []

    if valid_only:
        clauses.append("is_valid = 1")
    if question_type:
        clauses.append("question_type = ?")
        params.append(question_type)
    if min_quality is not None:
        clauses.append("quality_score >= ?")
        params.append(min_quality)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM questions {where_sql} ORDER BY id ASC"

    with get_db_connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def delete_questions_by_document(doc_id: int, db_path: Optional[str] = None) -> int:
    """Delete all questions associated with a document."""
    sql = "DELETE FROM questions WHERE document_id = ?"
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(sql, (doc_id,))
        return cursor.rowcount


def get_question_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Return question breakdown statistics (counts by type, valid vs invalid, math)."""
    with get_db_connection(db_path) as conn:
        try:
            total = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
            valid = conn.execute("SELECT COUNT(*) FROM questions WHERE is_valid = 1").fetchone()[0]
            mcq = conn.execute("SELECT COUNT(*) FROM questions WHERE question_type = 'mcq'").fetchone()[0]
            numerical = conn.execute("SELECT COUNT(*) FROM questions WHERE question_type = 'numerical'").fetchone()[0]
            short_ans = conn.execute("SELECT COUNT(*) FROM questions WHERE question_type = 'short_answer'").fetchone()[0]
            long_ans = conn.execute("SELECT COUNT(*) FROM questions WHERE question_type = 'long_answer'").fetchone()[0]
            with_math = conn.execute("SELECT COUNT(*) FROM questions WHERE has_math = 1").fetchone()[0]
            avg_quality = conn.execute("SELECT AVG(quality_score) FROM questions").fetchone()[0] or 0.0

            return {
                "total_questions": total,
                "valid_questions": valid,
                "invalid_questions": total - valid,
                "mcq_count": mcq,
                "numerical_count": numerical,
                "short_answer_count": short_ans,
                "long_answer_count": long_ans,
                "math_count": with_math,
                "avg_quality_score": round(float(avg_quality), 1),
            }
        except sqlite3.OperationalError:
            return {
                "total_questions": 0,
                "valid_questions": 0,
                "invalid_questions": 0,
                "mcq_count": 0,
                "numerical_count": 0,
                "short_answer_count": 0,
                "long_answer_count": 0,
                "math_count": 0,
                "avg_quality_score": 0.0,
            }


# ─── Similarity Edges CRUD ─────────────────────────────────────────

def insert_similarity_edge(
    source_id: int,
    target_id: int,
    layer: str,
    score: float,
    has_conflict: bool = False,
    conflict_note: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """
    Insert a similarity edge record between two questions.
    """
    sql = """
        INSERT INTO similarity_edges (
            source_question_id, target_question_id, similarity_layer, score,
            has_answer_conflict, conflict_note
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_question_id, target_question_id, similarity_layer) DO UPDATE SET
            score = excluded.score,
            has_answer_conflict = excluded.has_answer_conflict,
            conflict_note = excluded.conflict_note
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            sql,
            (source_id, target_id, layer, score, 1 if has_conflict else 0, conflict_note),
        )
        return cursor.lastrowid


def list_similarity_edges(
    layer: Optional[str] = None,
    conflict_only: bool = False,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List similarity edges with question text and document details joined."""
    clauses = []
    params: List[Any] = []

    if layer:
        clauses.append("e.similarity_layer = ?")
        params.append(layer)
    if conflict_only:
        clauses.append("e.has_answer_conflict = 1")

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    sql = f"""
        SELECT
            e.id,
            e.source_question_id,
            e.target_question_id,
            e.similarity_layer,
            e.score,
            e.has_answer_conflict,
            e.conflict_note,
            e.created_at,
            q1.question_text AS source_text,
            q1.question_number AS source_qnum,
            q1.answer AS source_answer,
            d1.filename AS source_doc,
            d1.year AS source_year,
            q2.question_text AS target_text,
            q2.question_number AS target_qnum,
            q2.answer AS target_answer,
            d2.filename AS target_doc,
            d2.year AS target_year
        FROM similarity_edges e
        JOIN questions q1 ON e.source_question_id = q1.id
        JOIN documents d1 ON q1.document_id = d1.id
        JOIN questions q2 ON e.target_question_id = q2.id
        JOIN documents d2 ON q2.document_id = d2.id
        {where_sql}
        ORDER BY e.score DESC, e.id DESC
    """
    with get_db_connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def get_similar_questions_for_id(question_id: int, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all similar questions linked to a given question ID."""
    sql = """
        SELECT
            e.id,
            e.similarity_layer,
            e.score,
            e.has_answer_conflict,
            e.conflict_note,
            CASE WHEN e.source_question_id = ? THEN q2.id ELSE q1.id END AS matched_question_id,
            CASE WHEN e.source_question_id = ? THEN q2.question_text ELSE q1.question_text END AS matched_text,
            CASE WHEN e.source_question_id = ? THEN q2.answer ELSE q1.answer END AS matched_answer,
            CASE WHEN e.source_question_id = ? THEN d2.filename ELSE d1.filename END AS matched_doc,
            CASE WHEN e.source_question_id = ? THEN d2.year ELSE d1.year END AS matched_year
        FROM similarity_edges e
        JOIN questions q1 ON e.source_question_id = q1.id
        JOIN documents d1 ON q1.document_id = d1.id
        JOIN questions q2 ON e.target_question_id = q2.id
        JOIN documents d2 ON q2.document_id = d2.id
        WHERE e.source_question_id = ? OR e.target_question_id = ?
        ORDER BY e.score DESC
    """
    with get_db_connection(db_path) as conn:
        rows = conn.execute(
            sql,
            (question_id, question_id, question_id, question_id, question_id, question_id, question_id),
        ).fetchall()
        return [dict(r) for r in rows]


def delete_similarity_edges(db_path: Optional[str] = None) -> int:
    """Clear all similarity edges from the database."""
    sql = "DELETE FROM similarity_edges"
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(sql)
        return cursor.rowcount


def get_similarity_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Return counts of exact, near-exact, semantic duplicates and answer conflicts."""
    with get_db_connection(db_path) as conn:
        try:
            total_edges = conn.execute("SELECT COUNT(*) FROM similarity_edges").fetchone()[0]
            exact_count = conn.execute(
                "SELECT COUNT(*) FROM similarity_edges WHERE similarity_layer = 'exact'"
            ).fetchone()[0]
            near_exact_count = conn.execute(
                "SELECT COUNT(*) FROM similarity_edges WHERE similarity_layer = 'near_exact'"
            ).fetchone()[0]
            semantic_count = conn.execute(
                "SELECT COUNT(*) FROM similarity_edges WHERE similarity_layer = 'semantic'"
            ).fetchone()[0]
            conflict_count = conn.execute(
                "SELECT COUNT(*) FROM similarity_edges WHERE has_answer_conflict = 1"
            ).fetchone()[0]

            return {
                "total_duplicates": total_edges,
                "exact_duplicates": exact_count,
                "near_exact_duplicates": near_exact_count,
                "semantic_duplicates": semantic_count,
                "answer_conflicts": conflict_count,
            }
        except sqlite3.OperationalError:
            return {
                "total_duplicates": 0,
                "exact_duplicates": 0,
                "near_exact_duplicates": 0,
                "semantic_duplicates": 0,
                "answer_conflicts": 0,
            }


def get_database_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Return aggregated counts and statistics from the database."""
    with get_db_connection(db_path) as conn:
        total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        ingested_docs = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE status IN ('ingested', 'extracted', 'indexed')"
        ).fetchone()[0]
        extracted_docs = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE status IN ('extracted', 'indexed')"
        ).fetchone()[0]
        failed_docs = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE status = 'failed'"
        ).fetchone()[0]
        total_pages = conn.execute("SELECT COUNT(*) FROM document_pages").fetchone()[0]
        ocr_pages = conn.execute(
            "SELECT COUNT(*) FROM document_pages WHERE ocr_applied = 1"
        ).fetchone()[0]

        q_stats = get_question_stats(db_path=db_path)
        sim_stats = get_similarity_stats(db_path=db_path)

        return {
            "total_documents": total_docs,
            "ingested_documents": ingested_docs,
            "extracted_documents": extracted_docs,
            "failed_documents": failed_docs,
            "total_pages": total_pages,
            "ocr_pages": ocr_pages,
            **q_stats,
            **sim_stats,
        }


# ─── Study Progress CRUD (SM-2 Spaced Repetition) ───────────────────

def save_study_progress(
    question_id: int,
    status: str,
    easiness_factor: float = 2.5,
    interval_days: int = 1,
    repetition_count: int = 0,
    quality_score: int = 0,
    next_review_date: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """
    Insert or update SM-2 study progress for a question.
    """
    sql = """
        INSERT INTO study_progress (
            question_id, status, easiness_factor, interval_days, repetition_count,
            quality_score, next_review_date, last_reviewed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(question_id) DO UPDATE SET
            status = excluded.status,
            easiness_factor = excluded.easiness_factor,
            interval_days = excluded.interval_days,
            repetition_count = excluded.repetition_count,
            quality_score = excluded.quality_score,
            next_review_date = excluded.next_review_date,
            last_reviewed_at = CURRENT_TIMESTAMP
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            sql,
            (
                question_id,
                status,
                float(easiness_factor),
                int(interval_days),
                int(repetition_count),
                int(quality_score),
                next_review_date,
            ),
        )
        return cursor.lastrowid


def get_study_progress(db_path: Optional[str] = None) -> Dict[int, Dict[str, Any]]:
    """
    Retrieve all study progress records indexed by question_id.
    """
    sql = "SELECT * FROM study_progress"
    with get_db_connection(db_path) as conn:
        try:
            rows = conn.execute(sql).fetchall()
            return {row["question_id"]: dict(row) for row in rows}
        except sqlite3.OperationalError:
            return {}


def get_study_progress_for_question(
    question_id: int,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve study progress for a specific question.
    """
    sql = "SELECT * FROM study_progress WHERE question_id = ?"
    with get_db_connection(db_path) as conn:
        try:
            row = conn.execute(sql, (question_id,)).fetchone()
            return dict(row) if row else None
        except sqlite3.OperationalError:
            return None


def get_study_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Return statistics on study progress (confident, needs_review, mastered, not_started).
    """
    with get_db_connection(db_path) as conn:
        try:
            total_tracked = conn.execute("SELECT COUNT(*) FROM study_progress").fetchone()[0]
            confident = conn.execute(
                "SELECT COUNT(*) FROM study_progress WHERE status = 'confident'"
            ).fetchone()[0]
            needs_review = conn.execute(
                "SELECT COUNT(*) FROM study_progress WHERE status = 'needs_review'"
            ).fetchone()[0]
            mastered = conn.execute(
                "SELECT COUNT(*) FROM study_progress WHERE status = 'mastered'"
            ).fetchone()[0]
            not_started = conn.execute(
                "SELECT COUNT(*) FROM study_progress WHERE status = 'not_started'"
            ).fetchone()[0]

            return {
                "total_tracked": total_tracked,
                "confident": confident,
                "needs_review": needs_review,
                "mastered": mastered,
                "not_started": not_started,
            }
        except sqlite3.OperationalError:
            return {
                "total_tracked": 0,
                "confident": 0,
                "needs_review": 0,
                "mastered": 0,
                "not_started": 0,
            }
