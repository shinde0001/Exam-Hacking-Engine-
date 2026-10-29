# Exam-Hacking Engine 🎯

> **Local, zero-API exam intelligence and question prediction platform.**
> Ingest past papers & books $\rightarrow$ Extract structured questions $\rightarrow$ Detect duplicates & answer conflicts $\rightarrow$ Discover exam setter patterns $\rightarrow$ Predict upcoming exam questions using 6 specialized algorithms $\rightarrow$ Generate spaced repetition study schedules.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Key Features by Phase](#key-features-by-phase)
  - [Phase 1: Ingestion & Document Processing](#phase-1-ingestion--document-processing)
  - [Phase 2: Question Extraction & Validation](#phase-2-question-extraction--validation)
  - [Phase 3: 3-Layer Similarity & Deduplication](#phase-3-3-layer-similarity--deduplication)
  - [Phase 4: Pattern Intelligence & Syllabus Mapping](#phase-4-pattern-intelligence--syllabus-mapping)
  - [Phase 5: 6-Algorithm Prediction Engine & Ensemble](#phase-5-6-algorithm-prediction-engine--ensemble)
  - [Phase 6: SM-2 Spaced Repetition Study Planner](#phase-6-sm-2-spaced-repetition-study-planner)
  - [Phase 7: Web UI & Multi-Format Exporter](#phase-7-web-ui--multi-format-exporter)
- [Tech Stack](#tech-stack)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Web Application](#running-the-web-application)
- [Running Automated Tests](#running-automated-tests)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Zero-API & Privacy Guarantee](#zero-api--privacy-guarantee)

---

## 🔍 Overview

The **Exam-Hacking Engine** is a self-hosted, local platform engineered to analyze past examination papers, syllabus documents, and textbooks without sending proprietary material to third-party cloud APIs.

By leveraging statistical modeling, symbolic mathematics, vector search, graph centrality, and machine learning, it extracts questions, eliminates duplicate items across multiple book editions, models examiner habits, and generates customized study plans with predicted mock examination papers.

---

## 🚀 Key Features by Phase

### Phase 1: Ingestion & Document Processing
- **Multi-Format Parsing**: Extracts text from PDF (`pdfplumber`, `PyMuPDF`, `pdftotext`), DOCX (`python-docx` with direct XML fallback), TXT, and scanned images.
- **OCR Fallback**: Automatically invokes Tesseract OCR when extracted characters fall below 50 characters per page.
- **Incremental Processing**: Tracks state per document (`pending` $\rightarrow$ `ingested` / `failed`) and skips identical files via deterministic SHA-256 hashing.
- **Metadata Detection**: Automatically extracts examination years from filenames (e.g., `Physics_2023_Final.pdf` $\rightarrow$ `2023`).

### Phase 2: Question Extraction & Validation
- **Finite-State Boundary Parser**: Detects question delimiters (`Q1`, `Question 1.`, `(1)`, `1)`).
- **Auto Question-Type Classification**:
  - `mcq`: Identifies multiple-choice options (`(A)`, `(B)`, `(C)`, `(D)`), option texts, and answer keys.
  - `numerical`: Detects calculation stems ("calculate", "find", "determine").
  - `short_answer`: 1–2 mark definitions and statements ("define", "state", "list").
  - `long_answer`: 5+ mark descriptive questions ("explain", "derive", "prove").
- **SymPy Symbolic Math Normalization**:
  - Unifies unicode exponents and symbols ($x^2 = x² = x^{**}2$, $\sqrt{x}$, $\frac{a}{b}$).
  - Performs structural and algebraic equivalence checks using SymPy.
- **Quality Scoring (0–100)**: Assesses questions based on stem length, MCQ option completeness, answer presence, and OCR artifact penalties. Questions with quality score $\ge 60$ proceed to prediction pipelines.

### Phase 3: 3-Layer Similarity & Deduplication
- **Layer 1 (Exact)**: SHA-256 hash on normalized text detects verbatim repetitions.
- **Layer 2 (Near-Exact)**: `rapidfuzz` token sort ratio ($\ge 85\%$) catches reworded stems.
- **Layer 3 (Semantic)**: Vector cosine similarity ($\ge 0.82$) identifies conceptually identical questions expressed in different phrasing.
- **FAISS Vector Index**: Fast nearest-neighbor retrieval using `IndexFlatIP` and `IndexIVFFlat` for 10k+ scale.
- **Answer Conflict Detection**: Automatically flags duplicate questions sourced from different papers that have conflicting answer keys.

### Phase 4: Pattern Intelligence & Syllabus Mapping
- **7 Core Statistical Patterns**:
  1. *Topic Frequency*: TF-IDF extraction and total marks weightage.
  2. *Year-over-Year Trends*: Time-series trajectory per topic across historical exam cycles.
  3. *Option-Position Bias*: Pearson's Chi-squared ($\chi^2$) test evaluating whether question setters disproportionately favor options A, B, C, or D.
  4. *Distractor Analysis*: Identifies distractor patterns (e.g. longest option bias, opposite-sign numerical traps).
  5. *Question-Setter Habits*: K-Means clustering on stylistic and lexical features.
  6. *Concept Co-occurrence*: Co-occurrence matrix identifying topics that consistently appear in the same paper.
  7. *Difficulty Tiering*: Composite score calculated from word count, vocabulary rarity, math complexity, and cognitive tier.
- **Bloom's Taxonomy Classification**: Tags cognitive hierarchy (*Remember*, *Understand*, *Apply*, *Analyze*, *Evaluate*, *Create*).
- **Syllabus Mapper**: Parses syllabus outlines (Markdown, plain text, or JSON) and maps questions to chapters using hybrid vector and keyword matching.

### Phase 5: 6-Algorithm Prediction Engine & Ensemble
Combines 6 specialized algorithms into a single weighted ensemble score:

| Algorithm | Method & Formula | Purpose |
|---|---|---|
| **1. Bayesian Probability** | Beta conjugate prior updated with recency-weighted likelihood | Frequency-based arrival probability |
| **2. Markov Chain** | Transition matrix $T_{ij} = P(\text{Topic}_j^{(n+1)} \mid \text{Topic}_i^{(n)})$ | Year-over-year topic rotation dynamics |
| **3. Poisson Process** | Arrival rate $P(k \ge 1) = 1 - e^{-\lambda}$ | Occurrence rate per examination cycle |
| **4. Cyclic Detection** | Standard deviation of gap intervals | Identifies periodic $N$-year cycles and `DUE_NOW` topics |
| **5. Association Rules** | Apriori co-occurrence boosts | Cross-topic appearance boosts |
| **6. PageRank Centrality** | Random-walk centrality on similarity graph | Uncovers foundational, central concept questions |

- **Confidence Badging**:
  - 🔴 **Very High Confidence**: $\ge 5$ algorithms agree
  - 🟡 **Medium Confidence**: $\ge 3$ algorithms agree
  - 🟢 **Low Confidence**: $< 3$ algorithms agree
- **Predicted Mock Paper Generator**: Generates balanced mock examination papers stratified by topic weightages and question types.
- **Minimum Study Set**: Greedy set-cover optimization providing maximum topic coverage with the fewest questions.

### Phase 6: SM-2 Spaced Repetition Study Planner
- **SuperMemo SM-2 Algorithm**: Dynamically adjusts Easiness Factor ($EF$), repetition counts, and intervals (1, 3, 7, 14, 30 days).
- **Adaptive Time Budgeting**: Estimates study time per question based on type, marks, and difficulty tier.
- **Day-by-Day Study Schedule**: Distributes initial learning and scheduled reviews across days remaining until the target examination date.
- **Progress Tracking**: Records student recall ratings (0–5) and updates statuses (`not_started`, `needs_review`, `confident`, `mastered`).

### Phase 7: Web UI & Multi-Format Exporter
- **Interactive UI**: Responsive Flask interface for uploads, question viewing, duplicate reviews, prediction insights, mock paper builder, and study schedules.
- **Export Formats**:
  - 📄 **PDF Mock Exam Papers**: Publication-ready, printable mock exam papers with instructions and marking schemes via ReportLab.
  - 📱 **Anki Flashcards**: Tab-separated `.txt` deck with styled cards, option formatting, and tags.
  - 📊 **CSV Question Bank**: Full corpus export including prediction metrics, Bloom tiers, and confidence badges.
  - 📅 **CSV Study Plan**: Day-by-day study tasks with estimated completion times.

---

## 🛠 Tech Stack

- **Language**: Python 3.10+
- **Web Framework**: Flask 3.1
- **Database**: SQLite (embedded, zero-configuration)
- **Document Ingestion**: `pdfplumber`, `PyMuPDF` (fitz), `python-docx`, `pdftotext`, `pytesseract`, `Pillow`
- **Math Engine**: `sympy`
- **Vector Search & Embeddings**: `faiss-cpu`, `sentence-transformers` (`all-MiniLM-L6-v2`) with scikit-learn TF-IDF + SVD fallback
- **Text Similarity & Matching**: `rapidfuzz`
- **Statistical & ML Modeling**: `scikit-learn`, `scipy`, `numpy`
- **PDF Generation**: `reportlab`

---

## 📂 Directory Structure

```text
Exam-Hacking Engine/
├── app.py                      # Flask web application & REST routes
├── config.py                   # Central configuration & algorithm thresholds
├── requirements.txt            # Python dependencies
├── implementation_plan.md      # Architecture & specification document
├── README.md                   # Project documentation
├── database/
│   ├── schema.sql              # SQLite schema definition
│   ├── exam_engine.db          # SQLite persistent database
│   ├── vector_index.faiss      # Persisted FAISS index
│   └── vector_mapping.json     # Question ID to vector mapping
├── engine/
│   ├── __init__.py
│   ├── db.py                   # Database access & query layer
│   ├── ingest.py               # Document ingestion pipeline & OCR
│   ├── extractor.py            # Question boundary & type parser
│   ├── math_parser.py          # SymPy symbolic math normalizer
│   ├── validator.py            # Question quality scoring & validator
│   ├── similarity.py           # 3-layer similarity & conflict detector
│   ├── vectorstore.py          # Vector store & FAISS search
│   ├── patterns.py             # 7 statistical pattern analyzers
│   ├── topic_model.py          # NMF & LDA topic modeling
│   ├── syllabus_mapper.py      # Syllabus parser & chapter mapper
│   ├── bloom_classifier.py     # Bloom's taxonomy cognitive tiering
│   ├── predictor.py            # 6 prediction algorithms & ensemble
│   ├── study_scheduler.py      # SM-2 spaced repetition planner
│   └── exporter.py             # PDF, Anki, and CSV export generators
├── templates/                  # Jinja2 HTML templates
├── static/                     # CSS stylesheets and client scripts
├── exports/                    # Generated PDF papers and export files
├── uploads/                    # Uploaded past papers and source files
└── tests/                      # Automated test suite (77 tests)
    ├── test_db.py
    ├── test_ingest.py
    ├── test_extractor.py
    ├── test_math_parser.py
    ├── test_validator.py
    ├── test_similarity.py
    ├── test_vectorstore.py
    ├── test_patterns.py
    ├── test_topic_model.py
    ├── test_syllabus_mapper.py
    ├── test_bloom_classifier.py
    ├── test_predictor.py
    ├── test_study_scheduler.py
    ├── test_exporter.py
    └── test_app.py
```

---

## 🏁 Getting Started

### Prerequisites
- Linux / macOS / Windows with WSL
- Python 3.10+
- Optional: `tesseract-ocr` for scanned image ingestion:
  ```bash
  sudo apt-get install tesseract-ocr
  ```

### Installation
1. Clone or navigate to the project directory:
   ```bash
   cd "Exam-Hacking Engine"
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Web Application
Start the Flask development server:
```bash
python3 app.py
```
Or when using vendored packages:
```bash
PYTHONPATH=.packages:. python3 app.py
```

Open your browser at:
```text
http://localhost:5000
```

---

## 🧪 Running Automated Tests

The engine includes 77 unit tests covering all 7 phases:

To run the entire test suite:
```bash
PYTHONPATH=.packages:. python3 -m unittest discover tests
```

To run tests for a specific phase:
```bash
# Phase 1: Ingestion & Database
PYTHONPATH=.packages:. python3 -m unittest tests/test_db.py tests/test_ingest.py

# Phase 2: Extraction, Math Normalization & Validation
PYTHONPATH=.packages:. python3 -m unittest tests/test_extractor.py tests/test_math_parser.py tests/test_validator.py

# Phase 3: Similarity & Vector Store
PYTHONPATH=.packages:. python3 -m unittest tests/test_similarity.py tests/test_vectorstore.py

# Phase 4: Pattern Intelligence & Topic Modeling
PYTHONPATH=.packages:. python3 -m unittest tests/test_patterns.py tests/test_topic_model.py tests/test_syllabus_mapper.py tests/test_bloom_classifier.py

# Phase 5: 6-Algorithm Prediction Engine
PYTHONPATH=.packages:. python3 -m unittest tests/test_predictor.py

# Phase 6: SM-2 Study Scheduler
PYTHONPATH=.packages:. python3 -m unittest tests/test_study_scheduler.py

# Phase 7: Web Application & Exporter
PYTHONPATH=.packages:. python3 -m unittest tests/test_app.py tests/test_exporter.py
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | `GET` | System diagnostic status & library availability |
| `/api/documents` | `GET` | List all ingested documents |
| `/api/documents/<id>` | `GET` | Get metadata and raw page texts for a document |
| `/api/documents/<id>/questions` | `GET` | Get extracted questions for a document |
| `/api/questions` | `GET` | Filter questions by type, quality score, or validity |
| `/api/search` | `GET` | Semantic search (`?q=<query>&top_k=10&min_score=0.4`) |
| `/api/duplicates` | `GET` | List duplicate question edges and answer conflicts |
| `/api/patterns` | `GET` | Complete 7-pattern intelligence report |
| `/api/syllabus` | `POST` | Map questions to syllabus text and compute chapter weightages |
| `/api/predictions` | `GET` | Ranked questions with 6-algorithm breakdown and confidence badges |
| `/api/predictions/mock-paper` | `POST` | Generate balanced predicted mock paper JSON |
| `/api/predictions/minimum-study-set` | `GET` | Minimum set-cover questions for full syllabus coverage |
| `/api/study-plan` | `GET / POST` | Day-by-day SM-2 spaced repetition study schedule |
| `/api/study-plan/record-review` | `POST` | Record student recall rating (0–5) and update SM-2 interval |
| `/export/questions/csv` | `GET` | Download question bank and prediction scores as CSV |
| `/export/questions/anki` | `GET` | Download Anki flashcard deck (`.txt` TSV) |
| `/export/mock-paper/pdf` | `GET` | Download printable mock exam paper PDF |
| `/export/study-plan/csv` | `GET` | Download day-by-day study calendar as CSV |

---

## ⚙️ Configuration

Key parameters in [`config.py`](file:///home/parth/Exam-Hacking%20Engine/config.py) can be customized:

```python
# Ingestion OCR threshold
OCR_CHAR_THRESHOLD = 50

# Similarity thresholds
SIMILARITY_FUZZY_THRESHOLD = 85        # rapidfuzz token_sort_ratio
SIMILARITY_SEMANTIC_THRESHOLD = 0.82   # Cosine similarity on embeddings

# Validation thresholds
MIN_QUESTION_LENGTH = 10
QUALITY_SCORE_THRESHOLD = 60

# Prediction Ensemble Weights (must sum to 1.0)
PREDICTION_WEIGHTS = {
    "bayesian": 0.25,
    "markov": 0.15,
    "poisson": 0.20,
    "cyclic": 0.15,
    "association": 0.10,
    "pagerank": 0.15,
}

# Spaced Repetition Intervals (days)
SPACED_REPETITION_INTERVALS = [1, 3, 7, 14, 30]
```

---

## 🔒 Zero-API & Privacy Guarantee

- All text extraction, embeddings, mathematical simplifications, statistical analyses, and PDF generations run **100% locally on your machine**.
- No exam questions, solutions, or uploaded textbooks are ever transmitted over external networks or sent to commercial LLM APIs.
