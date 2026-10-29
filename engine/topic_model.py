"""
Exam-Hacking Engine — Topic Modeling
Discovers latent topic clusters across the question corpus using NMF or LDA from Scikit-Learn.
Features:
- TF-IDF feature extraction with n-gram support
- NMF (Non-negative Matrix Factorization) & LDA (Latent Dirichlet Allocation)
- Automatic topic extraction and representative keyword labeling
- Assigns primary topic ID and keyword tags to each question
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import config

try:
    from sklearn.decomposition import NMF, LatentDirichletAllocation
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class TopicDefinition:
    topic_id: int
    name: str
    top_keywords: List[str]
    question_count: int


@dataclass
class QuestionTopicAssignment:
    question_id: int
    topic_id: int
    topic_name: str
    probability: float


@dataclass
class TopicModelResult:
    topics: List[TopicDefinition]
    assignments: List[QuestionTopicAssignment]
    method: str


def clean_text_for_topic_modeling(text: str) -> str:
    """Clean question text by removing non-alphabetic noise and math symbols."""
    t = text.lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def run_topic_modeling(
    questions: List[Dict[str, Any]],
    num_topics: Optional[int] = None,
    method: Optional[str] = None,
    top_n_words: int = 5,
) -> TopicModelResult:
    """
    Fit topic model on question texts and return topic definitions and question assignments.
    """
    total_q = len(questions)
    if total_q == 0 or not HAS_SKLEARN:
        return TopicModelResult(topics=[], assignments=[], method="none")

    # Adaptively scale topic count based on available questions
    target_topics = num_topics or config.NUM_TOPICS
    n_topics = max(2, min(target_topics, total_q // 2)) if total_q >= 4 else 1

    model_type = (method or config.TOPIC_MODEL_METHOD).lower()

    # Pre-clean texts
    corpus = [clean_text_for_topic_modeling(q.get("question_text", "")) for q in questions]

    # Filter out empty texts
    non_empty_indices = [i for i, t in enumerate(corpus) if len(t.split()) >= 2]
    if len(non_empty_indices) < 2:
        # Fallback for single or minimal questions
        single_topic = TopicDefinition(
            topic_id=0,
            name="General Questions",
            top_keywords=["general", "exam", "questions"],
            question_count=total_q,
        )
        assignments = [
            QuestionTopicAssignment(
                question_id=q["id"],
                topic_id=0,
                topic_name="General Questions",
                probability=1.0,
            )
            for q in questions
        ]
        return TopicModelResult(topics=[single_topic], assignments=assignments, method="fallback")

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_df=0.95,
            min_df=1,
            max_features=2000,
            ngram_range=(1, 2),
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()

        if model_type == "lda":
            model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=15,
                learning_method="online",
            )
        else:
            model = NMF(
                n_components=n_topics,
                random_state=42,
                max_iter=200,
                init="nndsvda",
            )

        doc_topic_dist = model.fit_transform(tfidf_matrix)

        # Extract top words per topic
        topics: List[TopicDefinition] = []
        topic_counts = {t_id: 0 for t_id in range(n_topics)}
        assignments: List[QuestionTopicAssignment] = []

        # Find dominant topic for each question
        for i, q in enumerate(questions):
            probs = doc_topic_dist[i]
            top_topic_id = int(np.argmax(probs))
            norm_factor = float(np.sum(probs))
            confidence = float(probs[top_topic_id] / norm_factor) if norm_factor > 0 else 1.0 / n_topics

            topic_counts[top_topic_id] += 1
            assignments.append(
                QuestionTopicAssignment(
                    question_id=q["id"],
                    topic_id=top_topic_id,
                    topic_name="",  # filled after naming
                    probability=round(confidence, 3),
                )
            )

        # Build topic names from top keywords
        for t_idx, topic_weights in enumerate(model.components_):
            top_indices = topic_weights.argsort()[-top_n_words:][::-1]
            keywords = [feature_names[i] for i in top_indices if topic_weights[i] > 0]
            if not keywords:
                keywords = [f"concept_{t_idx + 1}"]

            name = ", ".join(keywords[:3]).title()
            topics.append(
                TopicDefinition(
                    topic_id=t_idx,
                    name=name,
                    top_keywords=keywords,
                    question_count=topic_counts[t_idx],
                )
            )

        # Update topic names in assignments
        topic_name_map = {t.topic_id: t.name for t in topics}
        for a in assignments:
            a.topic_name = topic_name_map.get(a.topic_id, "General")

        return TopicModelResult(
            topics=topics,
            assignments=assignments,
            method=model_type,
        )

    except Exception as e:
        print(f"[TopicModel] Topic modeling failed: {e}. Falling back to single cluster.")
        fallback_topic = TopicDefinition(
            topic_id=0,
            name="General Concepts",
            top_keywords=["exam", "concepts"],
            question_count=total_q,
        )
        fallback_assignments = [
            QuestionTopicAssignment(
                question_id=q["id"],
                topic_id=0,
                topic_name="General Concepts",
                probability=1.0,
            )
            for q in questions
        ]
        return TopicModelResult(
            topics=[fallback_topic],
            assignments=fallback_assignments,
            method="fallback",
        )
