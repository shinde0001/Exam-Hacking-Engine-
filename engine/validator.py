"""
Exam-Hacking Engine — Question Validator
Performs quality scoring and validation checks on extracted questions.
Checks:
- Minimum length (stem < 10 chars -> severe extraction failure)
- MCQ options validation (expects 4 options A/B/C/D)
- Answer detection check
- OCR artifact scanning (|, ¬, §, ©, ¢, ¦, noise glyphs)
- Quality score (0 - 100) and threshold filtering (score >= 60 to enter pipeline)
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple
import config
from engine.extractor import ExtractedQuestion

# Regex for common OCR artifacts and noise glyphs
OCR_ARTIFACT_PATTERN = re.compile(r"[¬§©¢¦¤¶®™±\[\]\|_]{2,}|[¬§¢¦¤¶]")


@dataclass
class ValidationResult:
    is_valid: bool
    quality_score: int
    issues: List[str] = field(default_factory=list)


def calculate_ocr_artifact_penalty(text: str) -> Tuple[int, List[str]]:
    """
    Check text for OCR noise/artifacts and return penalty and issue notes.
    """
    if not text:
        return 0, []

    issues = []
    penalty = 0

    artifacts_found = OCR_ARTIFACT_PATTERN.findall(text)
    if artifacts_found:
        count = len(artifacts_found)
        if count >= 3:
            penalty = 25
            issues.append(f"Significant OCR artifacts detected ({count} occurrences: {', '.join(set(artifacts_found[:3]))})")
        else:
            penalty = 10
            issues.append(f"Minor OCR artifacts detected ({count} occurrences)")

    # Check excessive non-alphanumeric ratio (excluding whitespace and common punctuation)
    total_len = len(text)
    if total_len > 0:
        alnum_count = sum(1 for c in text if c.isalnum())
        ratio = alnum_count / total_len
        if ratio < 0.45:
            penalty += 15
            issues.append(f"Low alphanumeric character density ({ratio:.1%})")

    return penalty, issues


def validate_question(q: ExtractedQuestion) -> ValidationResult:
    """
    Compute quality score (0-100) for an extracted question.
    Only questions with quality_score >= config.QUALITY_SCORE_THRESHOLD (60)
    enter the downstream similarity and prediction pipelines.
    """
    issues: List[str] = []
    score = 70  # Baseline health score

    stem = q.question_text.strip()
    stem_len = len(stem)

    # 1. Critical Check: Length < MIN_QUESTION_LENGTH
    if stem_len < config.MIN_QUESTION_LENGTH:
        issues.append(f"Question text is too short ({stem_len} chars, minimum {config.MIN_QUESTION_LENGTH})")
        # Hard cap on short stems
        return ValidationResult(
            is_valid=False,
            quality_score=min(20, stem_len * 2),
            issues=issues,
        )
    elif stem_len < 20:
        score -= 5
        issues.append(f"Short question stem ({stem_len} chars)")
    else:
        # Healthy length bonus
        score += 5

    # 2. MCQ Validation
    if q.question_type == "mcq":
        num_options = len(q.options)
        if num_options == config.EXPECTED_MCQ_OPTIONS:
            score += 15  # Perfect MCQ structure
            # Check labels are standard A, B, C, D
            labels = [opt.get("label", "").upper() for opt in q.options]
            if labels == ["A", "B", "C", "D"]:
                score += 5
        elif num_options == 0:
            score -= 30
            issues.append("Question classified as MCQ but no options extracted")
        else:
            score -= 15
            issues.append(
                f"MCQ has {num_options} options (expected {config.EXPECTED_MCQ_OPTIONS})"
            )

        # Check for empty option text
        empty_opts = [opt.get("label") for opt in q.options if not opt.get("text", "").strip()]
        if empty_opts:
            score -= 15
            issues.append(f"Options with empty text: {', '.join(empty_opts)}")

    # 3. Answer Check
    if q.answer:
        score += 10  # Bonus for answer key detected
    else:
        if q.question_type == "mcq":
            score -= 5
            issues.append("No answer key detected for MCQ")

    # 4. OCR Artifacts Check
    combined_text = stem + " " + " ".join(opt.get("text", "") for opt in q.options)
    artifact_penalty, artifact_issues = calculate_ocr_artifact_penalty(combined_text)
    score -= artifact_penalty
    issues.extend(artifact_issues)

    # 5. Math syntax balance check (if math detected)
    if q.has_math:
        # Check parentheses and braces balance
        for open_ch, close_ch in (("(", ")"), ("{", "}"), ("[", "]")):
            if combined_text.count(open_ch) != combined_text.count(close_ch):
                score -= 10
                issues.append(f"Unbalanced math brackets '{open_ch}' vs '{close_ch}'")
                break

    # Clamp final score between 0 and 100
    final_score = max(0, min(100, score))
    is_valid = final_score >= config.QUALITY_SCORE_THRESHOLD

    return ValidationResult(
        is_valid=is_valid,
        quality_score=final_score,
        issues=issues,
    )
