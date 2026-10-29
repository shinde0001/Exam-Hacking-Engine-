"""
Exam-Hacking Engine — Syllabus Mapper
Maps extracted questions to user-defined syllabus chapters and sub-topics.
Features:
- Parses syllabus files (text, markdown, JSON) into structured chapters
- Hybrid matching (keyword matching + vector embedding similarity)
- Computes chapter weightage (marks percentage and question frequency)
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from engine import similarity, vectorstore


@dataclass
class ChapterDefinition:
    chapter_number: int
    title: str
    subtopics: List[str] = field(default_factory=list)

    @property
    def full_description(self) -> str:
        """Combined title and sub-topics for text embedding."""
        sub_str = ", ".join(self.subtopics) if self.subtopics else ""
        return f"{self.title}. {sub_str}".strip()


@dataclass
class QuestionChapterMapping:
    question_id: int
    chapter_number: int
    chapter_title: str
    match_score: float
    marks: int


@dataclass
class ChapterWeightage:
    chapter_number: int
    title: str
    subtopics: List[str]
    question_count: int
    total_marks: int
    weightage_percent: float
    mapped_questions: List[Dict[str, Any]] = field(default_factory=list)


def parse_syllabus_text(syllabus_text: str) -> List[ChapterDefinition]:
    """
    Parse syllabus text or markdown into ChapterDefinitions.
    Supports formats:
    - JSON array: [{"chapter": 1, "title": "...", "subtopics": [...]}]
    - Markdown / Text:
      Chapter 1: Kinematics
      - Velocity, acceleration, projectile motion
      Chapter 2: Thermodynamics
      - First law, heat, work, entropy
    """
    if not syllabus_text or not syllabus_text.strip():
        return []

    # 1. Try JSON
    stripped = syllabus_text.strip()
    if (stripped.startswith("[") and stripped.endswith("]")) or (stripped.startswith("{") and stripped.endswith("}")):
        try:
            data = json.loads(stripped)
            if isinstance(data, dict) and "chapters" in data:
                data = data["chapters"]
            if isinstance(data, list):
                chapters = []
                for idx, item in enumerate(data, start=1):
                    ch_num = item.get("chapter", item.get("chapter_number", idx))
                    title = item.get("title", item.get("name", f"Chapter {ch_num}"))
                    subtopics = item.get("subtopics", item.get("topics", []))
                    if isinstance(subtopics, str):
                        subtopics = [s.strip() for s in subtopics.split(",") if s.strip()]
                    chapters.append(ChapterDefinition(chapter_number=int(ch_num), title=title, subtopics=subtopics))
                if chapters:
                    return chapters
        except Exception:
            pass

    # 2. Text / Markdown Parser
    chapters: List[ChapterDefinition] = []
    lines = syllabus_text.splitlines()
    current_chapter: Optional[ChapterDefinition] = None
    chapter_counter = 1

    chapter_pattern = re.compile(
        r"^(?:(?:Chapter|Unit|Module|Section)\s*(\d+)[\:\.\-\)]?\s*|\b(\d+)[\.\:]\s*)(.*)",
        re.IGNORECASE,
    )

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        match = chapter_pattern.match(line_str)
        if match:
            # Save previous chapter
            if current_chapter is not None:
                chapters.append(current_chapter)

            num_str = match.group(1) or match.group(2)
            ch_num = int(num_str) if num_str and num_str.isdigit() else chapter_counter
            title = (match.group(3) or f"Chapter {ch_num}").strip()

            current_chapter = ChapterDefinition(
                chapter_number=ch_num,
                title=title,
                subtopics=[],
            )
            chapter_counter += 1
        elif current_chapter is not None:
            # Sub-topics (bullet points or comma-separated)
            sub_line = re.sub(r"^[\*\-\•\d+\.]+\s*", "", line_str).strip()
            if sub_line:
                parts = [p.strip() for p in re.split(r"[,;]", sub_line) if p.strip()]
                current_chapter.subtopics.extend(parts)

    if current_chapter is not None:
        chapters.append(current_chapter)

    # Fallback if no chapter headers found
    if not chapters:
        # Split by lines as separate chapters
        for idx, line in enumerate(lines, start=1):
            if line.strip():
                chapters.append(ChapterDefinition(chapter_number=idx, title=line.strip(), subtopics=[]))

    return chapters


def map_questions_to_syllabus(
    questions: List[Dict[str, Any]],
    chapters: List[ChapterDefinition],
) -> List[ChapterWeightage]:
    """
    Map questions to syllabus chapters using hybrid matching:
    1. Keyword overlap with chapter title & subtopics
    2. Vector cosine similarity with chapter descriptions
    Returns list of ChapterWeightage objects with aggregated metrics.
    """
    if not chapters:
        return []

    store = vectorstore.get_vector_store()

    # Pre-encode chapter descriptions
    chapter_texts = [ch.full_description for ch in chapters]
    chapter_vectors = store.encode(chapter_texts)

    # Initialize chapter containers
    weightage_map: Dict[int, ChapterWeightage] = {
        ch.chapter_number: ChapterWeightage(
            chapter_number=ch.chapter_number,
            title=ch.title,
            subtopics=ch.subtopics,
            question_count=0,
            total_marks=0,
            weightage_percent=0.0,
            mapped_questions=[],
        )
        for ch in chapters
    }

    if not questions:
        return list(weightage_map.values())

    total_corpus_marks = 0

    for q in questions:
        q_text = q.get("question_text", "")
        q_norm = similarity.normalize_text_for_comparison(q_text)
        q_words = set(q_norm.split())
        marks = int(q.get("marks") or 1)
        total_corpus_marks += marks

        # Encode question vector
        q_vec = store.encode([q_text])[0]

        best_score = -1.0
        best_chapter_num = chapters[0].chapter_number

        for idx, ch in enumerate(chapters):
            # Keyword score
            ch_words = set(similarity.normalize_text_for_comparison(ch.full_description).split())
            overlap = len(q_words & ch_words)
            keyword_score = min(1.0, overlap / 4.0) if q_words else 0.0

            # Vector cosine similarity
            if len(chapter_vectors) > idx:
                vec_score = float(np.dot(q_vec, chapter_vectors[idx]))
            else:
                vec_score = 0.0

            # Hybrid score (60% vector, 40% keyword)
            hybrid_score = 0.60 * max(0.0, vec_score) + 0.40 * keyword_score

            if hybrid_score > best_score:
                best_score = hybrid_score
                best_chapter_num = ch.chapter_number

        # Assign question to best chapter
        target_ch = weightage_map[best_chapter_num]
        target_ch.question_count += 1
        target_ch.total_marks += marks
        target_ch.mapped_questions.append({
            **q,
            "match_score": round(best_score, 3),
        })

    # Compute percentage weightage
    results = list(weightage_map.values())
    for ch in results:
        if total_corpus_marks > 0:
            ch.weightage_percent = round((ch.total_marks / total_corpus_marks) * 100.0, 1)
        else:
            ch.weightage_percent = 0.0

    # Sort chapters by weightage descending
    results.sort(key=lambda x: x.total_marks, reverse=True)
    return results
