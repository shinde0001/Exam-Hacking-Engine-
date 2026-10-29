"""
Exam-Hacking Engine — Central Configuration
All settings, paths, thresholds, and model parameters in one place.
"""

import os
import sys

# ─── Base Paths ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGES_DIR = os.path.join(BASE_DIR, ".packages")
if os.path.isdir(PACKAGES_DIR) and PACKAGES_DIR not in sys.path:
    sys.path.insert(0, PACKAGES_DIR)

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "exam_engine.db")

# Auto-create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)

# ─── Flask Settings ──────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "exam-hacking-engine-dev-secret-key-replace-in-prod")
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB max upload
ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "txt", "png", "jpg", "jpeg", "tiff", "bmp"}

# ─── Ingestion Settings ──────────────────────────────────────
# If extracted text has fewer chars than this per page, trigger OCR
OCR_CHAR_THRESHOLD = 50

# Tesseract OCR configuration
TESSERACT_CONFIG = "--oem 3 --psm 6"  # LSTM engine, assume uniform block of text

# ─── NLP & Embedding Models ──────────────────────────────────
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

# ─── Similarity Thresholds ───────────────────────────────────
SIMILARITY_FUZZY_THRESHOLD = 85        # rapidfuzz token_sort_ratio (0-100)
SIMILARITY_SEMANTIC_THRESHOLD = 0.82   # cosine similarity (0.0-1.0)
SIMILARITY_TOP_K = 20                  # FAISS neighbours to check per question

# ─── Question Validation ─────────────────────────────────────
MIN_QUESTION_LENGTH = 10               # chars — below this = extraction error
QUALITY_SCORE_THRESHOLD = 60           # 0-100 — below this = skip question
EXPECTED_MCQ_OPTIONS = 4               # flag if MCQ has != this many options

# ─── Topic Modelling ─────────────────────────────────────────
NUM_TOPICS = 25                        # LDA/NMF topic count (auto-tunable later)
TOPIC_MODEL_METHOD = "nmf"             # "lda" or "nmf"

# ─── Prediction Engine Weights ────────────────────────────────
# Ensemble weights for 6 prediction algorithms (must sum to 1.0)
PREDICTION_WEIGHTS = {
    "bayesian": 0.25,
    "markov": 0.15,
    "poisson": 0.20,
    "cyclic": 0.15,
    "association": 0.10,
    "pagerank": 0.15,
}

# Confidence thresholds
CONFIDENCE_HIGH = 0.80       # >= this = "Very High Confidence"
CONFIDENCE_MEDIUM = 0.50     # >= this = "Medium Confidence"
# below CONFIDENCE_MEDIUM    = "Low Confidence"

# ─── Cyclic Detection ────────────────────────────────────────
CYCLIC_STD_THRESHOLD = 1.0   # max std deviation of gaps to consider cyclic
CYCLIC_MAX_PERIOD = 5        # max years for a cycle

# ─── Study Scheduler ─────────────────────────────────────────
SPACED_REPETITION_INTERVALS = [1, 3, 7, 14, 30]  # days between reviews
DEFAULT_DAILY_STUDY_HOURS = 4

# ─── Export Settings ──────────────────────────────────────────
EXPORT_DIR = os.path.join(BASE_DIR, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)
