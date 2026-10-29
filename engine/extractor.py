"""
Exam-Hacking Engine — Question Extractor
Extracts structured questions from raw ingested text using regex and finite-state boundary parsing.
Features:
- Question boundary detection (Q1, Question 1, 1., etc.)
- Multi-format option extraction for MCQs (A/B/C/D)
- Marks extraction ([2 marks], (5 Marks), etc.)
- Answer key extraction (Ans: B, Answer: 42, etc.)
- Auto-classification into question types: MCQ, NUMERICAL, SHORT_ANSWER, LONG_ANSWER
- Mathematical expression tagging via engine.math_parser
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from engine import db, math_parser


@dataclass
class ExtractedQuestion:
    question_number: str
    question_text: str
    question_type: str  # 'mcq', 'numerical', 'short_answer', 'long_answer'
    options: List[Dict[str, str]] = field(default_factory=list)
    answer: Optional[str] = None
    marks: Optional[int] = None
    page_number: int = 1
    has_math: bool = False
    normalized_text: str = ""
    raw_block: str = ""


# Regex Patterns for Boundary Detection
QUESTION_START_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:"
    r"(?:Q(?:uestion)?\.?\s*(\d+[a-z]?|\([a-z\d]+\))[\.\:\)\-]?)"  # Question 1, Q1., Q.1), Q(a)
    r"|(?:\b(\d{1,3})[\.\)]\s+)"                                      # 1. , 2) (at least 1 whitespace)
    r"|(?:\((\d{1,3})\)\s+)"                                          # (1) , (2)
    r")",
    re.IGNORECASE,
)

# MCQ Option Start Patterns
# Matches: (A) , A. , A) , [A]
OPTION_START_PATTERN = re.compile(
    r"(?:^|[\s\n])(?:\(([A-Ea-e])\)|([A-Ea-e])[\.\)]|\[([A-Ea-e])\])\s+",
)

# Marks Indicator: [5 Marks], (2 marks), [10], (3 M)
MARKS_PATTERN = re.compile(
    r"(?:\[|\()(\d{1,2})\s*(?:marks?|mark|m|pts?|points?)?(?:\]|\))",
    re.IGNORECASE,
)

# Answer Indicator: Ans: B, Answer: (C), Solution: 42
ANSWER_PATTERN = re.compile(
    r"(?:^|\n|\s*)(?:Ans(?:wer)?|Solution|Key)\s*[\:\.\-]?\s*(?:\(([A-Da-d0-9\.\-\s]+)\)|([A-Da-d0-9\.\-\s\w]+))",
    re.IGNORECASE,
)

# Section Headers to Skip
SECTION_HEADER_PATTERN = re.compile(
    r"^\s*(?:SECTION|PART|GROUP|MODULE)\s+[A-Z0-9IVX]+",
    re.IGNORECASE,
)

# Type classification keywords
NUMERICAL_KEYWORDS = re.compile(
    r"\b(?:calculate|find|determine|evaluate|compute|solve for|how many|what is the value of)\b",
    re.IGNORECASE,
)
SHORT_KEYWORDS = re.compile(
    r"\b(?:define|state|list|name|give|what is|what are|mention|write down)\b",
    re.IGNORECASE,
)
LONG_KEYWORDS = re.compile(
    r"\b(?:explain|derive|prove|describe|discuss|elaborate|illustrate|distinguish between|compare and contrast|with neat diagram)\b",
    re.IGNORECASE,
)


def extract_marks(text: str) -> Tuple[Optional[int], str]:
    """
    Extract marks from text and return (marks, text_with_marks_stripped).
    """
    match = MARKS_PATTERN.search(text)
    if match:
        try:
            marks = int(match.group(1))
            cleaned = text[:match.start()] + text[match.end():]
            return marks, cleaned.strip()
        except (ValueError, IndexError):
            pass
    return None, text


def extract_answer(text: str) -> Tuple[Optional[str], str]:
    """
    Extract answer/solution substring from question block and return (answer, cleaned_text).
    """
    match = ANSWER_PATTERN.search(text)
    if match:
        ans = (match.group(1) or match.group(2) or "").strip()
        cleaned = text[:match.start()] + text[match.end():]
        return ans if ans else None, cleaned.strip()
    return None, text


def extract_mcq_options(text: str) -> Tuple[List[Dict[str, str]], str]:
    """
    Extract MCQ options (A, B, C, D) and return (options_list, question_stem).
    Slices option text accurately even if options contain internal brackets/parentheses.
    """
    matches = list(OPTION_START_PATTERN.finditer(text))
    if len(matches) < 2:
        return [], text

    labels = [(m.group(1) or m.group(2) or m.group(3)).upper() for m in matches]
    # Verify sequential start (first must be A, second must be B)
    if labels[0] != "A" or labels[1] != "B":
        return [], text

    first_opt_start = matches[0].start()
    stem = text[:first_opt_start].strip()

    options: List[Dict[str, str]] = []
    seen_labels = set()

    for i, m in enumerate(matches):
        label = labels[i]
        if label in seen_labels:
            continue

        start_idx = m.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        opt_text = text[start_idx:end_idx].strip()
        opt_text = re.sub(r"\s+", " ", opt_text)

        options.append({"label": label, "text": opt_text})
        seen_labels.add(label)

    return options, stem


def classify_question_type(
    stem: str,
    has_options: bool,
    marks: Optional[int] = None,
    has_numerical_answer: bool = False,
) -> str:
    """
    Automatically detect question type:
    - mcq: Has multiple choice options
    - numerical: Contains numerical calculation verbs and/or mathematical calculation
    - short_answer: "define/state/list" or 1-2 marks
    - long_answer: "explain/derive/prove" or >= 5 marks
    """
    if has_options:
        return "mcq"

    stem_lower = stem.lower()

    if marks is not None:
        if marks >= 5:
            return "long_answer"
        elif marks <= 2 and not NUMERICAL_KEYWORDS.search(stem):
            return "short_answer"

    if LONG_KEYWORDS.search(stem):
        return "long_answer"

    if NUMERICAL_KEYWORDS.search(stem) or (math_parser.contains_math(stem) and "=" in stem):
        return "numerical"

    if SHORT_KEYWORDS.search(stem):
        return "short_answer"

    # Default based on length if no clear marker
    word_count = len(stem.split())
    if word_count > 30:
        return "long_answer"
    else:
        return "short_answer"


def split_text_into_question_blocks(raw_text: str) -> List[Tuple[str, str]]:
    """
    Splits continuous document text into (question_number, block_text) pairs.
    Uses regex boundary matching for question starts.
    """
    lines = raw_text.splitlines()
    blocks: List[Tuple[str, str]] = []
    current_q_num: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_lines:
                current_lines.append("")
            continue

        # Skip section headers
        if SECTION_HEADER_PATTERN.match(stripped):
            continue

        # Check for question boundary
        match = QUESTION_START_PATTERN.match(line)
        if match:
            # Found a new question start!
            if current_q_num is not None and current_lines:
                full_block = "\n".join(current_lines).strip()
                if full_block:
                    blocks.append((current_q_num, full_block))

            matched_num = match.group(1) or match.group(2) or match.group(3) or ""
            current_q_num = matched_num.strip("().:- ")
            # Start new block with text after the question number indicator
            text_after = line[match.end():].strip()
            current_lines = [text_after] if text_after else []
        else:
            if current_q_num is not None:
                current_lines.append(line)
            else:
                # Text before first question (e.g. instructions/header)
                pass

    if current_q_num is not None and current_lines:
        full_block = "\n".join(current_lines).strip()
        if full_block:
            blocks.append((current_q_num, full_block))

    return blocks


def parse_question_block(
    q_num: str,
    raw_block: str,
    page_number: int = 1,
) -> ExtractedQuestion:
    """
    Parses a single question text block into structured ExtractedQuestion.
    """
    text = raw_block

    # 1. Extract marks if present
    marks, text = extract_marks(text)

    # 2. Extract answer if present
    answer, text = extract_answer(text)

    # 3. Extract MCQ options if present
    options, stem = extract_mcq_options(text)

    # Clean stem whitespace
    stem = re.sub(r"\s+", " ", stem).strip()

    # 4. Check for mathematical content
    has_math = math_parser.contains_math(stem) or any(
        math_parser.contains_math(opt["text"]) for opt in options
    )
    normalized_math = math_parser.normalize_math_expression(stem) if has_math else stem

    # 5. Classify question type
    has_numerical_ans = bool(answer and re.search(r"\b\d+(?:\.\d+)?\b", answer))
    q_type = classify_question_type(
        stem=stem,
        has_options=len(options) > 0,
        marks=marks,
        has_numerical_answer=has_numerical_ans,
    )

    return ExtractedQuestion(
        question_number=q_num,
        question_text=stem,
        question_type=q_type,
        options=options,
        answer=answer,
        marks=marks,
        page_number=page_number,
        has_math=has_math,
        normalized_text=normalized_math,
        raw_block=raw_block,
    )


def extract_questions_from_pages(
    pages: List[Dict[str, Any]],
) -> List[ExtractedQuestion]:
    """
    Extract structured questions across document pages with page tracking.
    """
    questions: List[ExtractedQuestion] = []
    # Join pages while maintaining page offset markers
    for p in pages:
        page_num = p["page_number"]
        raw_text = p["raw_text"]
        blocks = split_text_into_question_blocks(raw_text)

        for q_num, block in blocks:
            extracted = parse_question_block(
                q_num=q_num,
                raw_block=block,
                page_number=page_num,
            )
            questions.append(extracted)

    return questions


def extract_and_store_document_questions(
    doc_id: int,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    High-level extraction pipeline for a document:
    1. Loads raw pages from database.
    2. Runs question extraction.
    3. Runs quality validation (engine.validator).
    4. Persists questions into the database.
    5. Updates document status to 'extracted'.
    """
    from engine import validator

    pages = db.get_document_pages(doc_id, db_path=db_path)
    if not pages:
        raise ValueError(f"No pages found for document ID {doc_id}. Please ingest the document first.")

    raw_questions = extract_questions_from_pages(pages)
    stored_questions = []

    # Clear old questions if re-extracting
    db.delete_questions_by_document(doc_id, db_path=db_path)

    for q in raw_questions:
        val_result = validator.validate_question(q)

        # Store in database
        q_id = db.insert_question(
            doc_id=doc_id,
            page_number=q.page_number,
            question_number=q.question_number,
            question_text=q.question_text,
            question_type=q.question_type,
            options_json=json.dumps(q.options) if q.options else None,
            answer=q.answer,
            marks=q.marks,
            quality_score=val_result.quality_score,
            is_valid=val_result.is_valid,
            validation_issues=json.dumps(val_result.issues),
            has_math=q.has_math,
            normalized_text=q.normalized_text,
            db_path=db_path,
        )

        stored_questions.append({
            "id": q_id,
            "question_number": q.question_number,
            "question_text": q.question_text,
            "question_type": q.question_type,
            "options": q.options,
            "answer": q.answer,
            "marks": q.marks,
            "quality_score": val_result.quality_score,
            "is_valid": val_result.is_valid,
            "issues": val_result.issues,
            "has_math": q.has_math,
            "page_number": q.page_number,
        })

    # Update document status to extracted
    db.update_document_status(doc_id, "extracted", db_path=db_path)

    return stored_questions
