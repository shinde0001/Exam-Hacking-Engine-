"""
Exam-Hacking Engine — Similarity & Deduplication Engine
Implements the 3-layer similarity and duplicate detection pipeline:
- Layer 1: Exact Match (SHA-256 hash on normalized text)
- Layer 2: Near-Exact Match (rapidfuzz token_sort_ratio >= 85)
- Layer 3: Semantic Match (Cosine similarity on vector embeddings >= 0.82)
- Answer Conflict Detection (Same question with conflicting answers across sources)
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import config
from engine import math_parser

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


@dataclass
class SimilarityResult:
    is_duplicate: bool
    layer: str  # 'exact', 'near_exact', 'semantic', 'none'
    score: float  # 0.0 to 1.0
    has_conflict: bool = False
    conflict_note: Optional[str] = None


def normalize_text_for_comparison(text: str) -> str:
    """
    Produce canonical text for similarity comparison:
    - Normalizes mathematical expressions (exponents, fractions, operators)
    - Lowercases text
    - Strips punctuation except math symbols (+, -, *, /, ^, =)
    - Collapses whitespace
    """
    if not text:
        return ""

    # Normalize math notation first
    text = math_parser.normalize_math_expression(text)
    text = text.lower()

    # Remove non-math punctuation
    text = re.sub(r"[^\w\s\+\-\*\/\^=]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_exact_hash(normalized_text: str) -> str:
    """Compute SHA-256 hash of normalized text for verbatim repeat detection."""
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def compute_fuzzy_ratio(norm1: str, norm2: str) -> float:
    """
    Compute token sort ratio between two strings using rapidfuzz (or fallback).
    Returns score normalized from 0.0 to 1.0.
    """
    if not norm1 or not norm2:
        return 0.0

    if HAS_RAPIDFUZZ:
        ratio = fuzz.token_sort_ratio(norm1, norm2)
        return round(ratio / 100.0, 4)

    # Pure Python token Jaccard fallback if rapidfuzz is unavailable
    tokens1 = set(norm1.split())
    tokens2 = set(norm2.split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return round(intersection / union, 4)


def check_answer_conflict(ans1: Optional[str], ans2: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Check if two duplicate questions have conflicting answers.
    Returns (has_conflict, conflict_note).
    """
    if not ans1 or not ans2:
        return False, None

    clean1 = ans1.strip().upper()
    clean2 = ans2.strip().upper()

    # If exactly equal
    if clean1 == clean2:
        return False, None

    # Strip parentheses e.g. "(B)" -> "B"
    clean1_strip = re.sub(r"[\(\)\[\]]", "", clean1).strip()
    clean2_strip = re.sub(r"[\(\)\[\]]", "", clean2).strip()
    if clean1_strip == clean2_strip:
        return False, None

    # Check mathematical equivalence if math is present
    if math_parser.contains_math(ans1) or math_parser.contains_math(ans2):
        if math_parser.are_math_expressions_equivalent(ans1, ans2):
            return False, None

    return True, f"Conflicting answers detected: '{ans1}' vs '{ans2}'"


def compare_question_pair(
    q1_text: str,
    q2_text: str,
    ans1: Optional[str] = None,
    ans2: Optional[str] = None,
    semantic_score: Optional[float] = None,
) -> SimilarityResult:
    """
    Evaluate two questions against the 3-layer similarity model.
    Checks in sequence:
    1. Exact Match: SHA-256 on normalized text
    2. Near-Exact Match: Rapidfuzz token sort ratio >= threshold (85%)
    3. Semantic Match: Cosine similarity >= threshold (0.82)
    """
    norm1 = normalize_text_for_comparison(q1_text)
    norm2 = normalize_text_for_comparison(q2_text)

    # Layer 1: Exact Match
    if norm1 and norm2 and compute_exact_hash(norm1) == compute_exact_hash(norm2):
        has_conf, conf_note = check_answer_conflict(ans1, ans2)
        return SimilarityResult(
            is_duplicate=True,
            layer="exact",
            score=1.0,
            has_conflict=has_conf,
            conflict_note=conf_note,
        )

    # Layer 2: Near-Exact Match
    fuzzy_threshold = config.SIMILARITY_FUZZY_THRESHOLD / 100.0  # e.g. 0.85
    fuzzy_score = compute_fuzzy_ratio(norm1, norm2)
    if fuzzy_score >= fuzzy_threshold:
        has_conf, conf_note = check_answer_conflict(ans1, ans2)
        return SimilarityResult(
            is_duplicate=True,
            layer="near_exact",
            score=fuzzy_score,
            has_conflict=has_conf,
            conflict_note=conf_note,
        )

    # Layer 3: Semantic Match
    if semantic_score is not None and semantic_score >= config.SIMILARITY_SEMANTIC_THRESHOLD:
        has_conf, conf_note = check_answer_conflict(ans1, ans2)
        return SimilarityResult(
            is_duplicate=True,
            layer="semantic",
            score=round(semantic_score, 4),
            has_conflict=has_conf,
            conflict_note=conf_note,
        )

    return SimilarityResult(
        is_duplicate=False,
        layer="none",
        score=max(fuzzy_score, semantic_score or 0.0),
        has_conflict=False,
        conflict_note=None,
    )
