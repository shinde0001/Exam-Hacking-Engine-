-- Exam-Hacking Engine — SQLite Database Schema
-- Defines core tables for Document Ingestion and prepares future schemas

PRAGMA foreign_keys = ON;

-- 1. Documents (tracked for incremental processing and status)
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE,
    file_type TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    year INTEGER,
    subject TEXT,
    total_pages INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending', -- pending, ingesting, ingested, extracted, indexed, failed
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Document Pages (stores raw extracted text & OCR details per page)
CREATE TABLE IF NOT EXISTS document_pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    char_count INTEGER NOT NULL DEFAULT 0,
    extraction_method TEXT NOT NULL,
    ocr_applied INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE,
    UNIQUE(document_id, page_number)
);

-- 3. Extracted Questions (structured question items with options, type, marks, quality)
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    question_number TEXT,
    question_text TEXT NOT NULL,
    question_type TEXT NOT NULL, -- mcq, numerical, short_answer, long_answer
    options_json TEXT,           -- JSON array of options: [{"label": "A", "text": "..."}]
    answer TEXT,
    marks INTEGER,
    quality_score INTEGER NOT NULL DEFAULT 0,
    is_valid INTEGER NOT NULL DEFAULT 1,
    validation_issues TEXT,      -- JSON array of issue strings
    has_math INTEGER NOT NULL DEFAULT 0,
    normalized_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
);

-- 4. Similarity Edges (stores detected exact, near-exact, and semantic duplicates + answer conflicts)
CREATE TABLE IF NOT EXISTS similarity_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_question_id INTEGER NOT NULL,
    target_question_id INTEGER NOT NULL,
    similarity_layer TEXT NOT NULL, -- exact, near_exact, semantic
    score REAL NOT NULL,
    has_answer_conflict INTEGER NOT NULL DEFAULT 0,
    conflict_note TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_question_id) REFERENCES questions (id) ON DELETE CASCADE,
    FOREIGN KEY (target_question_id) REFERENCES questions (id) ON DELETE CASCADE,
    UNIQUE(source_question_id, target_question_id, similarity_layer)
);

-- 5. Study Progress (SM-2 spaced repetition tracking per question)
CREATE TABLE IF NOT EXISTS study_progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'not_started', -- not_started, needs_review, confident, mastered
    easiness_factor REAL DEFAULT 2.5,
    interval_days INTEGER DEFAULT 1,
    repetition_count INTEGER DEFAULT 0,
    quality_score INTEGER DEFAULT 0,
    next_review_date TEXT,
    last_reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (question_id) REFERENCES questions (id) ON DELETE CASCADE
);

-- Indexes for fast query lookup
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_document_pages_doc ON document_pages(document_id);
CREATE INDEX IF NOT EXISTS idx_document_pages_doc_page ON document_pages(document_id, page_number);
CREATE INDEX IF NOT EXISTS idx_questions_doc ON questions(document_id);
CREATE INDEX IF NOT EXISTS idx_questions_type ON questions(question_type);
CREATE INDEX IF NOT EXISTS idx_questions_valid ON questions(is_valid);
CREATE INDEX IF NOT EXISTS idx_questions_quality ON questions(quality_score);
CREATE INDEX IF NOT EXISTS idx_sim_source ON similarity_edges(source_question_id);
CREATE INDEX IF NOT EXISTS idx_sim_target ON similarity_edges(target_question_id);
CREATE INDEX IF NOT EXISTS idx_sim_layer ON similarity_edges(similarity_layer);
CREATE INDEX IF NOT EXISTS idx_sim_conflict ON similarity_edges(has_answer_conflict);
CREATE INDEX IF NOT EXISTS idx_study_status ON study_progress(status);
CREATE INDEX IF NOT EXISTS idx_study_next_review ON study_progress(next_review_date);
