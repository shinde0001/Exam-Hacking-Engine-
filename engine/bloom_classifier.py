"""
Exam-Hacking Engine — Bloom's Taxonomy Classifier
Classifies exam questions into the 6 cognitive domains of Bloom's Revised Taxonomy:
1. Remember: Recall facts and basic concepts (define, state, list, name)
2. Understand: Explain ideas or concepts (explain, describe, summarize, discuss)
3. Apply: Use information in new situations (calculate, solve, compute, determine)
4. Analyze: Draw connections among ideas (compare, contrast, distinguish, differentiate)
5. Evaluate: Justify a stand or decision (assess, critique, judge, justify)
6. Create: Produce new or original work (design, construct, develop, derive, formulate)
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

BLOOM_LEVELS = [
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
]

BLOOM_LEVEL_WEIGHTS = {
    "remember": 1,
    "understand": 2,
    "apply": 3,
    "analyze": 4,
    "evaluate": 5,
    "create": 6,
}

# Regex Keyword Heuristics for each Bloom Level
BLOOM_KEYWORD_PATTERNS = {
    "create": re.compile(
        r"\b(?:design|construct|develop|derive|formulate|compose|devise|generate|propose|synthesize|plan|build)\b",
        re.IGNORECASE,
    ),
    "evaluate": re.compile(
        r"\b(?:justify|assess|critique|judge|defend|verify|recommend|prioritize|evaluate the validity|appraise)\b",
        re.IGNORECASE,
    ),
    "analyze": re.compile(
        r"\b(?:compare|contrast|distinguish|differentiate|examine|relate|categorize|separate|deduce|discriminate|break down)\b",
        re.IGNORECASE,
    ),
    "apply": re.compile(
        r"\b(?:calculate|solve|determine|compute|find|demonstrate|implement|use|apply|show that|calculate the value)\b",
        re.IGNORECASE,
    ),
    "understand": re.compile(
        r"\b(?:explain|describe|summarize|interpret|classify|discuss|clarify|give reasons|why does|how does|illustrate)\b",
        re.IGNORECASE,
    ),
    "remember": re.compile(
        r"\b(?:define|state|list|name|recall|identify|mention|write down|what is|what are|which of the following)\b",
        re.IGNORECASE,
    ),
}


@dataclass
class BloomClassification:
    level: str  # One of BLOOM_LEVELS
    tier: int   # 1 to 6
    confidence: float
    matched_keywords: List[str]


def classify_bloom_level(
    question_text: str,
    question_type: Optional[str] = None,
    has_math: bool = False,
) -> BloomClassification:
    """
    Classify a question into Bloom's taxonomy.
    Evaluates keyword matches from higher-order (create/evaluate) to foundational (remember).
    """
    if not question_text:
        return BloomClassification(level="remember", tier=1, confidence=0.5, matched_keywords=[])

    text_lower = question_text.lower()
    matches_by_level: Dict[str, List[str]] = {}

    for level, pattern in BLOOM_KEYWORD_PATTERNS.items():
        found = pattern.findall(text_lower)
        if found:
            matches_by_level[level] = list(set(found))

    # Priority 1: Check matches in hierarchical order (from highest order down)
    for level in ["create", "evaluate", "analyze", "apply", "understand", "remember"]:
        if level in matches_by_level:
            matched = matches_by_level[level]
            # Multiple keyword match boosts confidence
            confidence = min(0.95, 0.70 + 0.10 * len(matched))
            return BloomClassification(
                level=level,
                tier=BLOOM_LEVEL_WEIGHTS[level],
                confidence=round(confidence, 2),
                matched_keywords=matched,
            )

    # Priority 2: Fallback based on question type & mathematical content
    if question_type == "numerical" or (has_math and "=" in question_text):
        return BloomClassification(level="apply", tier=3, confidence=0.75, matched_keywords=["(numerical calculation)"])

    if question_type == "mcq":
        return BloomClassification(level="remember", tier=1, confidence=0.60, matched_keywords=["(mcq recognition)"])

    if question_type == "long_answer":
        return BloomClassification(level="understand", tier=2, confidence=0.65, matched_keywords=["(long answer explanation)"])

    return BloomClassification(level="remember", tier=1, confidence=0.50, matched_keywords=["(default baseline)"])


def compute_bloom_distribution(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate Bloom's taxonomy statistics across a corpus of questions.
    Returns counts, percentages, and average cognitive tier.
    """
    counts = {level: 0 for level in BLOOM_LEVELS}
    total = len(questions)

    if total == 0:
        return {
            "total_questions": 0,
            "counts": counts,
            "percentages": {level: 0.0 for level in BLOOM_LEVELS},
            "average_cognitive_tier": 0.0,
            "dominant_level": "remember",
        }

    tier_sum = 0
    for q in questions:
        text = q.get("question_text", "")
        q_type = q.get("question_type")
        has_math = bool(q.get("has_math"))

        res = classify_bloom_level(text, question_type=q_type, has_math=has_math)
        counts[res.level] += 1
        tier_sum += res.tier

    percentages = {
        level: round((count / total) * 100.0, 1) for level, count in counts.items()
    }

    dominant_level = max(counts, key=counts.get)
    avg_tier = round(tier_sum / total, 2)

    return {
        "total_questions": total,
        "counts": counts,
        "percentages": percentages,
        "average_cognitive_tier": avg_tier,
        "dominant_level": dominant_level,
    }
