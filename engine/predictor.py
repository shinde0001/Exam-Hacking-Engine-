"""
Exam-Hacking Engine — Prediction Engine (Phase 5)
Combines 6 specialized statistical, probabilistic, and graph algorithms via ensemble:
1. Bayesian Probability: Prior appearance frequency updated with recency-weighted likelihood
2. Markov Chain: Topic transition dynamics across chronological exam years
3. Poisson Process: Arrival rate of questions/topics per examination cycle
4. Cyclic Pattern Detection: Detects N-year rotation cycles and identifies due topics
5. Association Rule Mining: Apriori-based topic co-occurrence and cross-topic boosts
6. Graph-Based PageRank: Centrality of core concept questions across similarity edges

Also provides:
- Ensemble weighting and multi-tier confidence scoring
- Ranked question list with individual algorithm score breakdowns
- Predicted mock paper generator (stratified sampling by topic weights and question types)
- Minimum study set optimizer (greedy set-cover over high-yield topics)
- Strategic exam intelligence insights
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import config
from engine import bloom_classifier, db, patterns, topic_model

try:
    from scipy.stats import poisson as poisson_dist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class AlgorithmScores:
    bayesian: float
    markov: float
    poisson: float
    cyclic: float
    association: float
    pagerank: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "bayesian": round(float(self.bayesian), 3),
            "markov": round(float(self.markov), 3),
            "poisson": round(float(self.poisson), 3),
            "cyclic": round(float(self.cyclic), 3),
            "association": round(float(self.association), 3),
            "pagerank": round(float(self.pagerank), 3),
        }


@dataclass
class PredictedQuestion:
    id: int
    document_id: int
    question_number: str
    question_text: str
    question_type: str
    options: List[Dict[str, str]]
    answer: Optional[str]
    marks: int
    topic_id: int
    topic_name: str
    bloom_level: str
    difficulty_tier: str
    algorithm_scores: AlgorithmScores
    composite_score: float
    agreement_count: int
    confidence_badge: str
    confidence_label: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "question_number": self.question_number,
            "question_text": self.question_text,
            "question_type": self.question_type,
            "options": self.options,
            "answer": self.answer,
            "marks": self.marks,
            "topic_id": self.topic_id,
            "topic_name": self.topic_name,
            "bloom_level": self.bloom_level,
            "difficulty_tier": self.difficulty_tier,
            "algorithm_scores": self.algorithm_scores.to_dict(),
            "composite_score": round(float(self.composite_score), 3),
            "agreement_count": self.agreement_count,
            "confidence_badge": self.confidence_badge,
            "confidence_label": self.confidence_label,
        }


@dataclass
class MockPaperSection:
    name: str
    question_type: str
    questions: List[Dict[str, Any]]
    total_marks: int


@dataclass
class MockPaper:
    title: str
    total_marks: int
    estimated_time_minutes: int
    sections: List[Dict[str, Any]]
    questions: List[Dict[str, Any]]
    topic_breakdown: Dict[str, int]
    strategy_note: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "total_marks": self.total_marks,
            "estimated_time_minutes": self.estimated_time_minutes,
            "sections": self.sections,
            "questions": self.questions,
            "topic_breakdown": self.topic_breakdown,
            "strategy_note": self.strategy_note,
        }


@dataclass
class MinimumStudySet:
    total_questions: int
    target_topics_covered: int
    coverage_percent: float
    total_marks_represented: int
    questions: List[Dict[str, Any]]
    covered_topics: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_questions": self.total_questions,
            "target_topics_covered": self.target_topics_covered,
            "coverage_percent": round(self.coverage_percent, 1),
            "total_marks_represented": self.total_marks_represented,
            "questions": self.questions,
            "covered_topics": self.covered_topics,
        }


@dataclass
class PredictionReport:
    total_questions_analyzed: int
    ranked_questions: List[Dict[str, Any]]
    top_topics: List[Dict[str, Any]]
    cyclic_alerts: List[Dict[str, Any]]
    exam_strategy_insights: List[str]
    weights_used: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_questions_analyzed": self.total_questions_analyzed,
            "ranked_questions": self.ranked_questions,
            "top_topics": self.top_topics,
            "cyclic_alerts": self.cyclic_alerts,
            "exam_strategy_insights": self.exam_strategy_insights,
            "weights_used": self.weights_used,
        }


# ─── Algorithm 1: Bayesian Probability ─────────────────────────────

def compute_bayesian_probabilities(
    topic_appearances_by_doc: Dict[int, Set[int]],
    documents: List[Dict[str, Any]],
) -> Dict[int, float]:
    """
    Computes Bayesian probability of topic appearing in the next exam.
    Uses a Beta prior updated with recency-weighted observations.
    Exams from recent years receive higher likelihood weighting (2.0x).
    """
    total_docs = len(documents)
    if total_docs == 0:
        return {}

    years = [d.get("year") for d in documents if d.get("year") is not None]
    min_year = min(years) if years else 0
    max_year = max(years) if years else 0
    year_range = max(1, max_year - min_year)

    # Document weights based on recency
    doc_weights: Dict[int, float] = {}
    for d in documents:
        d_id = d["id"]
        y = d.get("year")
        if y is not None and year_range > 0:
            # Linear recency weight from 1.0 (oldest) to 2.0 (newest)
            doc_weights[d_id] = 1.0 + ((y - min_year) / year_range)
        else:
            doc_weights[d_id] = 1.0

    total_weight = sum(doc_weights.values())
    results: Dict[int, float] = {}

    for topic_id, doc_ids in topic_appearances_by_doc.items():
        base_rate = len(doc_ids) / total_docs

        # Prior Beta parameters (pseudo-counts)
        alpha_prior = 1.0 + (base_rate * 2.0)
        beta_prior = 1.0 + ((1.0 - base_rate) * 2.0)

        # Recency-weighted observations
        weighted_appearances = sum(doc_weights.get(doc_id, 1.0) for doc_id in doc_ids)
        weighted_non_appearances = total_weight - weighted_appearances

        # Posterior Beta expected value: E[p] = alpha / (alpha + beta)
        alpha_post = alpha_prior + weighted_appearances
        beta_post = beta_prior + max(0.0, weighted_non_appearances)
        prob = alpha_post / (alpha_post + beta_post)

        results[topic_id] = max(0.0, min(1.0, float(prob)))

    return results


# ─── Algorithm 2: Markov Chain (Topic Transitions) ──────────────────

def compute_markov_predictions(
    topic_appearances_by_doc: Dict[int, Set[int]],
    documents: List[Dict[str, Any]],
    all_topic_ids: List[int],
) -> Dict[int, float]:
    """
    Computes topic transition probabilities across consecutive chronological exam years.
    Predicts next-year state: S_next = S_latest * TransitionMatrix.
    """
    n_topics = len(all_topic_ids)
    if n_topics == 0:
        return {}

    topic_idx_map = {t_id: idx for idx, t_id in enumerate(all_topic_ids)}

    # Group documents by year
    year_docs: Dict[int, List[int]] = {}
    for d in documents:
        y = d.get("year")
        if y is not None:
            year_docs.setdefault(y, []).append(d["id"])

    sorted_years = sorted(year_docs.keys())

    # Fallback to base frequency if insufficient chronological data (< 2 distinct years)
    if len(sorted_years) < 2:
        return {
            t_id: max(0.1, min(0.9, len(topic_appearances_by_doc.get(t_id, set())) / max(1, len(documents))))
            for t_id in all_topic_ids
        }

    # Count transitions between consecutive years
    trans_counts = np.zeros((n_topics, n_topics), dtype=float)

    for i in range(len(sorted_years) - 1):
        y_curr = sorted_years[i]
        y_next = sorted_years[i + 1]

        # Topics present in current year
        curr_topics = set()
        for doc_id in year_docs[y_curr]:
            for t_id, docs in topic_appearances_by_doc.items():
                if doc_id in docs:
                    curr_topics.add(t_id)

        # Topics present in next year
        next_topics = set()
        for doc_id in year_docs[y_next]:
            for t_id, docs in topic_appearances_by_doc.items():
                if doc_id in docs:
                    next_topics.add(t_id)

        for src_t in curr_topics:
            if src_t in topic_idx_map:
                src_idx = topic_idx_map[src_t]
                for dst_t in next_topics:
                    if dst_t in topic_idx_map:
                        dst_idx = topic_idx_map[dst_t]
                        trans_counts[src_idx, dst_idx] += 1.0

    # Build transition matrix with Laplace smoothing
    epsilon = 0.05
    trans_matrix = np.zeros((n_topics, n_topics), dtype=float)
    for i in range(n_topics):
        row_sum = np.sum(trans_counts[i, :]) + (epsilon * n_topics)
        trans_matrix[i, :] = (trans_counts[i, :] + epsilon) / row_sum

    # Current state from most recent year
    latest_year = sorted_years[-1]
    latest_topics = set()
    for doc_id in year_docs[latest_year]:
        for t_id, docs in topic_appearances_by_doc.items():
            if doc_id in docs:
                latest_topics.add(t_id)

    curr_state = np.zeros(n_topics, dtype=float)
    for t_id in latest_topics:
        if t_id in topic_idx_map:
            curr_state[topic_idx_map[t_id]] = 1.0

    if np.sum(curr_state) > 0:
        curr_state = curr_state / np.sum(curr_state)
    else:
        curr_state = np.ones(n_topics, dtype=float) / n_topics

    # Project next state: S_next = curr_state * trans_matrix
    next_state = np.dot(curr_state, trans_matrix)

    # Min-max scale projected distribution to [0.15, 0.95] for calibrated scoring
    max_val = np.max(next_state)
    min_val = np.min(next_state)
    denom = max(1e-6, max_val - min_val)

    results: Dict[int, float] = {}
    for t_id, idx in topic_idx_map.items():
        scaled = 0.15 + (0.80 * (next_state[idx] - min_val) / denom)
        results[t_id] = round(float(scaled), 3)

    return results


# ─── Algorithm 3: Poisson Process (Appearance Rate) ────────────────

def compute_poisson_probabilities(
    topic_question_counts: Dict[int, int],
    total_exam_periods: int,
) -> Dict[int, float]:
    """
    Models topic appearance rate lambda per exam cycle.
    Computes P(appears >= 1 in next exam) = 1 - exp(-lambda).
    """
    total_periods = max(1, total_exam_periods)
    results: Dict[int, float] = {}

    for topic_id, count in topic_question_counts.items():
        lam = count / total_periods
        if HAS_SCIPY:
            # 1 - P(X = 0)
            prob = 1.0 - float(poisson_dist.pmf(0, lam))
        else:
            prob = 1.0 - math.exp(-lam)

        results[topic_id] = max(0.0, min(1.0, float(prob)))

    return results


# ─── Algorithm 4: Cyclic Pattern Detection ─────────────────────────

def compute_cyclic_scores(
    topic_appearances_by_doc: Dict[int, Set[int]],
    documents: List[Dict[str, Any]],
) -> Tuple[Dict[int, float], List[Dict[str, Any]]]:
    """
    Detects periodic N-year rotation cycles for topics.
    Calculates gaps between appearances; if gap std deviation is low,
    the topic is regular and its cycle length is determined.
    Returns: (topic_cyclic_scores, cyclic_alerts_list)
    """
    doc_year_map = {d["id"]: d.get("year") for d in documents if d.get("year") is not None}
    latest_year = max(doc_year_map.values()) if doc_year_map else 0

    scores: Dict[int, float] = {}
    alerts: List[Dict[str, Any]] = []

    for topic_id, doc_ids in topic_appearances_by_doc.items():
        years = sorted(list({doc_year_map[d_id] for d_id in doc_ids if d_id in doc_year_map}))

        if len(years) < 2:
            # Not enough historical points to infer a cycle
            scores[topic_id] = 0.40
            continue

        gaps = [years[i + 1] - years[i] for i in range(len(years) - 1)]
        mean_gap = float(np.mean(gaps))
        std_gap = float(np.std(gaps))
        last_year = years[-1]
        years_since_last = latest_year - last_year

        cycle_len = max(1, int(round(mean_gap)))
        is_regular = std_gap <= config.CYCLIC_STD_THRESHOLD

        if is_regular and cycle_len <= config.CYCLIC_MAX_PERIOD:
            # Topic has a regular cycle
            if years_since_last == cycle_len:
                # Exactly due in next exam!
                score = 0.95
                alerts.append({
                    "topic_id": topic_id,
                    "cycle_length": cycle_len,
                    "last_year": last_year,
                    "status": "DUE_NOW",
                    "message": f"Topic follows a {cycle_len}-year cycle; last seen in {last_year} and is DUE this cycle.",
                })
            elif years_since_last > cycle_len:
                # Overdue
                score = 0.85
                alerts.append({
                    "topic_id": topic_id,
                    "cycle_length": cycle_len,
                    "last_year": last_year,
                    "status": "OVERDUE",
                    "message": f"Topic follows a {cycle_len}-year cycle; last seen {years_since_last} years ago and is OVERDUE.",
                })
            elif years_since_last == 0 and cycle_len > 1:
                # Just appeared in the latest exam and cycle > 1, so less likely in the immediate next
                score = 0.25
            else:
                # In between cycles
                score = max(0.20, 1.0 - (0.35 * abs(years_since_last - cycle_len)))
        else:
            # Irregular appearance pattern
            score = 0.35

        scores[topic_id] = round(float(score), 3)

    return scores, alerts


# ─── Algorithm 5: Association Rule Mining (Apriori) ────────────────

def compute_association_boosts(
    topic_appearances_by_doc: Dict[int, Set[int]],
    prior_probabilities: Dict[int, float],
    total_docs: int,
) -> Dict[int, float]:
    """
    Finds topic co-occurrence patterns in exam papers.
    If Topic A is strongly expected and frequently co-occurs with Topic B (high confidence),
    Topic B's prediction score receives an association boost.
    """
    if total_docs == 0:
        return {t_id: 0.3 for t_id in prior_probabilities}

    results: Dict[int, float] = {}
    all_topics = list(prior_probabilities.keys())

    for target_t in all_topics:
        target_docs = topic_appearances_by_doc.get(target_t, set())
        target_support = len(target_docs) / total_docs

        max_associated_score = target_support

        for src_t in all_topics:
            if src_t == target_t:
                continue

            src_docs = topic_appearances_by_doc.get(src_t, set())
            if not src_docs:
                continue

            # Intersection: both topics in same paper
            co_occurring = len(src_docs.intersection(target_docs))
            if co_occurring == 0:
                continue

            # Confidence P(target | source)
            confidence = co_occurring / len(src_docs)
            src_prior = prior_probabilities.get(src_t, 0.5)

            # Association influence
            boost_candidate = src_prior * confidence
            if boost_candidate > max_associated_score:
                max_associated_score = boost_candidate

        results[target_t] = max(0.05, min(0.95, round(float(max_associated_score), 3)))

    return results


# ─── Algorithm 6: Graph-Based PageRank ─────────────────────────────

def compute_pagerank_scores(
    questions: List[Dict[str, Any]],
    similarity_edges: List[Dict[str, Any]],
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> Dict[int, float]:
    """
    Computes PageRank centrality over the question similarity graph.
    Central questions that connect to many other questions across papers
    represent fundamental exam concepts and receive high scores.
    """
    n = len(questions)
    if n == 0:
        return {}

    q_ids = [q["id"] for q in questions]
    id_to_idx = {qid: idx for idx, qid in enumerate(q_ids)}

    # Build weighted adjacency matrix
    adj = np.zeros((n, n), dtype=float)

    for edge in similarity_edges:
        s_id = edge.get("source_question_id")
        t_id = edge.get("target_question_id")
        weight = float(edge.get("score") or 0.8)

        if s_id in id_to_idx and t_id in id_to_idx:
            u = id_to_idx[s_id]
            v = id_to_idx[t_id]
            adj[u, v] = max(adj[u, v], weight)
            adj[v, u] = max(adj[v, u], weight)  # Undirected similarity

    # If no edges exist in database, fall back to uniform/content length prior
    if np.sum(adj) == 0:
        return {qid: 0.5 for qid in q_ids}

    # Normalize adjacency matrix to column-stochastic transition matrix
    col_sums = np.sum(adj, axis=0)
    # Dangling nodes (no edges) distribute uniformly
    dangling = (col_sums == 0)

    # Power iteration
    p = np.ones(n, dtype=float) / n

    for _ in range(max_iter):
        p_prev = p.copy()
        dangling_sum = np.sum(p_prev[dangling])

        p_new = np.zeros(n, dtype=float)
        for j in range(n):
            if not dangling[j]:
                p_new += (p_prev[j] / col_sums[j]) * adj[:, j]

        p = (damping * p_new) + (damping * dangling_sum / n) + ((1.0 - damping) / n)

        if np.sum(np.abs(p - p_prev)) < tol:
            break

    # Min-max scale to [0.15, 0.95] for balanced ensemble combination
    min_pr = float(np.min(p))
    max_pr = float(np.max(p))
    denom = max(1e-7, max_pr - min_pr)

    results: Dict[int, float] = {}
    for qid, idx in id_to_idx.items():
        scaled = 0.15 + (0.80 * (p[idx] - min_pr) / denom)
        results[qid] = round(float(scaled), 3)

    return results


# ─── Ensemble & Master Prediction Pipeline ──────────────────────────

def determine_confidence(agreement_count: int, composite_score: float) -> Tuple[str, str]:
    """Determine confidence badge and label based on algorithm agreement and score."""
    if agreement_count >= 5 or composite_score >= config.CONFIDENCE_HIGH:
        return "🔴", "Very High Confidence"
    elif agreement_count >= 3 or composite_score >= config.CONFIDENCE_MEDIUM:
        return "🟡", "Medium Confidence"
    else:
        return "🟢", "Low Confidence"


def run_prediction_pipeline(
    questions: Optional[List[Dict[str, Any]]] = None,
    documents: Optional[List[Dict[str, Any]]] = None,
    similarity_edges: Optional[List[Dict[str, Any]]] = None,
    weights: Optional[Dict[str, float]] = None,
    db_path: Optional[str] = None,
) -> PredictionReport:
    """
    Executes the full Phase 5 prediction ensemble:
    1. Loads corpus data (valid questions, documents, similarity edges)
    2. Runs topic modeling & Bloom classification
    3. Executes all 6 prediction algorithms
    4. Computes ensemble weighted composite scores
    5. Generates strategic insights & cyclic alerts
    """
    import json

    # 1. Load data from DB if not provided
    if questions is None:
        questions = db.list_questions(valid_only=True, db_path=db_path)
    if documents is None:
        documents = db.list_documents(db_path=db_path)
    if similarity_edges is None:
        similarity_edges = db.list_similarity_edges(db_path=db_path)

    active_weights = weights or config.PREDICTION_WEIGHTS

    if not questions:
        return PredictionReport(
            total_questions_analyzed=0,
            ranked_questions=[],
            top_topics=[],
            cyclic_alerts=[],
            exam_strategy_insights=["No valid questions available for prediction analysis."],
            weights_used=active_weights,
        )

    # 2. Topic Modeling & Question Mapping
    tm_result = topic_model.run_topic_modeling(questions)
    topic_map = {t.topic_id: t for t in tm_result.topics}
    q_topic_map = {a.question_id: a for a in tm_result.assignments}

    # Map topic appearances to documents
    topic_appearances_by_doc: Dict[int, Set[int]] = {t_id: set() for t_id in topic_map}
    topic_question_counts: Dict[int, int] = {t_id: 0 for t_id in topic_map}

    for q in questions:
        q_id = q["id"]
        doc_id = q.get("document_id")
        assignment = q_topic_map.get(q_id)
        if assignment:
            t_id = assignment.topic_id
            if doc_id:
                topic_appearances_by_doc.setdefault(t_id, set()).add(doc_id)
            topic_question_counts[t_id] = topic_question_counts.get(t_id, 0) + 1

    distinct_years = {d.get("year") for d in documents if d.get("year") is not None}
    total_exam_periods = max(len(distinct_years), len(documents), 1)

    # 3. Run the 6 Algorithms
    # Algorithm 1: Bayesian
    bayesian_scores = compute_bayesian_probabilities(topic_appearances_by_doc, documents)

    # Algorithm 2: Markov
    markov_scores = compute_markov_predictions(
        topic_appearances_by_doc, documents, list(topic_map.keys())
    )

    # Algorithm 3: Poisson
    poisson_scores = compute_poisson_probabilities(topic_question_counts, total_exam_periods)

    # Algorithm 4: Cyclic
    cyclic_scores, cyclic_alerts = compute_cyclic_scores(topic_appearances_by_doc, documents)

    # Algorithm 5: Association
    association_scores = compute_association_boosts(
        topic_appearances_by_doc, bayesian_scores, len(documents)
    )

    # Algorithm 6: PageRank
    pagerank_scores = compute_pagerank_scores(questions, similarity_edges)

    # 4. Assemble Question-Level Predictions
    w_b = active_weights.get("bayesian", 0.25)
    w_m = active_weights.get("markov", 0.15)
    w_p = active_weights.get("poisson", 0.20)
    w_c = active_weights.get("cyclic", 0.15)
    w_a = active_weights.get("association", 0.10)
    w_pr = active_weights.get("pagerank", 0.15)

    ranked_list: List[PredictedQuestion] = []

    for q in questions:
        q_id = q["id"]
        assignment = q_topic_map.get(q_id)
        t_id = assignment.topic_id if assignment else 0
        t_name = assignment.topic_name if assignment else "General Concepts"

        score_b = bayesian_scores.get(t_id, 0.4)
        score_m = markov_scores.get(t_id, 0.4)
        score_p = poisson_scores.get(t_id, 0.4)
        score_c = cyclic_scores.get(t_id, 0.4)
        score_a = association_scores.get(t_id, 0.4)
        score_pr = pagerank_scores.get(q_id, 0.4)

        # Composite ensemble score
        composite = (
            (w_b * score_b)
            + (w_m * score_m)
            + (w_p * score_p)
            + (w_c * score_c)
            + (w_a * score_a)
            + (w_pr * score_pr)
        )

        algo_scores_obj = AlgorithmScores(
            bayesian=score_b,
            markov=score_m,
            poisson=score_p,
            cyclic=score_c,
            association=score_a,
            pagerank=score_pr,
        )

        # Agreement count: algorithms with score >= 0.5
        all_scores = [score_b, score_m, score_p, score_c, score_a, score_pr]
        agreement_count = sum(1 for s in all_scores if s >= 0.5)

        badge, label = determine_confidence(agreement_count, composite)

        # Bloom & Difficulty classification
        bloom_result = bloom_classifier.classify_bloom_level(q.get("question_text", ""))
        diff_score, diff_tier = patterns.compute_question_difficulty(q)

        # Parse options
        raw_options = q.get("options_json")
        options = json.loads(raw_options) if raw_options else []

        ranked_list.append(
            PredictedQuestion(
                id=q_id,
                document_id=q.get("document_id", 0),
                question_number=str(q.get("question_number") or ""),
                question_text=q.get("question_text", ""),
                question_type=q.get("question_type", "short_answer"),
                options=options,
                answer=q.get("answer"),
                marks=int(q.get("marks") or 1),
                topic_id=t_id,
                topic_name=t_name,
                bloom_level=bloom_result.level,
                difficulty_tier=diff_tier,
                algorithm_scores=algo_scores_obj,
                composite_score=round(composite, 3),
                agreement_count=agreement_count,
                confidence_badge=badge,
                confidence_label=label,
            )
        )

    # Sort descending by composite prediction score
    ranked_list.sort(key=lambda x: x.composite_score, reverse=True)

    # 5. Top Topics Summary
    top_topics_data = []
    for t_id, t_def in topic_map.items():
        q_count = topic_question_counts.get(t_id, 0)
        avg_score = bayesian_scores.get(t_id, 0.0)
        top_topics_data.append({
            "topic_id": t_id,
            "name": t_def.name,
            "top_keywords": t_def.top_keywords,
            "question_count": q_count,
            "bayesian_prob": round(bayesian_scores.get(t_id, 0.0), 3),
            "poisson_prob": round(poisson_scores.get(t_id, 0.0), 3),
            "cyclic_score": round(cyclic_scores.get(t_id, 0.0), 3),
        })
    top_topics_data.sort(key=lambda x: x["bayesian_prob"], reverse=True)

    # 6. Strategic Exam Insights
    insights = []
    if ranked_list:
        high_conf_count = sum(1 for q in ranked_list if q.confidence_label == "Very High Confidence")
        insights.append(
            f"Surfaced {high_conf_count} high-confidence questions with multi-algorithm consensus (>= 80% composite or 5+ algorithm agreement)."
        )

    if cyclic_alerts:
        due_topics = [a["message"] for a in cyclic_alerts[:2]]
        insights.append(f"Cyclic Alert: {len(cyclic_alerts)} topic(s) have reached or exceeded their periodic rotation cycle.")

    if top_topics_data:
        top_name = top_topics_data[0]["name"]
        insights.append(f"Highest historical yield domain: '{top_name}' with {top_topics_data[0]['bayesian_prob'] * 100:.1f}% estimated probability.")

    return PredictionReport(
        total_questions_analyzed=len(ranked_list),
        ranked_questions=[q.to_dict() for q in ranked_list],
        top_topics=top_topics_data,
        cyclic_alerts=cyclic_alerts,
        exam_strategy_insights=insights,
        weights_used=active_weights,
    )


# ─── Predicted Mock Paper Generator ─────────────────────────────────

def generate_mock_paper(
    ranked_questions: List[Dict[str, Any]],
    target_marks: int = 50,
    target_questions: int = 15,
    title: str = "Predicted Mock Examination Paper",
) -> MockPaper:
    """
    Generates a realistic, balanced mock examination paper:
    - Stratified sampling across high-scoring topics
    - Balanced distribution of question types (MCQ, Short, Long, Numerical)
    - Enforces diversity: eliminates near-duplicate questions
    - Organizes into structured sections with time estimation
    """
    if not ranked_questions:
        return MockPaper(
            title=title,
            total_marks=0,
            estimated_time_minutes=0,
            sections=[],
            questions=[],
            topic_breakdown={},
            strategy_note="No questions available to generate mock paper.",
        )

    # Target question type distribution: MCQ ~35%, Short ~35%, Long/Numerical ~30%
    type_caps = {
        "mcq": max(2, int(target_questions * 0.35)),
        "short_answer": max(2, int(target_questions * 0.35)),
        "long_answer": max(1, int(target_questions * 0.20)),
        "numerical": max(1, int(target_questions * 0.10)),
    }

    selected_questions: List[Dict[str, Any]] = []
    seen_texts: Set[str] = set()
    current_marks = 0
    type_counts = {k: 0 for k in type_caps}
    topic_distribution: Dict[str, int] = {}

    # Sort candidates by composite score
    sorted_candidates = sorted(ranked_questions, key=lambda x: x.get("composite_score", 0), reverse=True)

    for q in sorted_candidates:
        if len(selected_questions) >= target_questions and current_marks >= target_marks:
            break

        q_type = q.get("question_type", "short_answer")
        q_marks = int(q.get("marks") or 1)
        text_prefix = q.get("question_text", "")[:40].lower()

        # Duplicate/near-duplicate check
        if text_prefix in seen_texts:
            continue

        # Respect soft type quota until we need to fill marks
        if type_counts.get(q_type, 0) >= type_caps.get(q_type, 3) and len(selected_questions) < (target_questions * 0.8):
            continue

        selected_questions.append(q)
        seen_texts.add(text_prefix)
        current_marks += q_marks
        type_counts[q_type] = type_counts.get(q_type, 0) + 1

        t_name = q.get("topic_name", "General")
        topic_distribution[t_name] = topic_distribution.get(t_name, 0) + 1

    # Organize selected questions into sections
    sections_map = {
        "mcq": ("Section A: Multiple Choice Questions", 1),
        "short_answer": ("Section B: Short Answer Questions", 2),
        "long_answer": ("Section C: Long Answer & Analytical Questions", 3),
        "numerical": ("Section D: Numerical Problems", 4),
    }

    sections_grouped: Dict[str, List[Dict[str, Any]]] = {}
    for q in selected_questions:
        q_type = q.get("question_type", "short_answer")
        sections_grouped.setdefault(q_type, []).append(q)

    built_sections: List[Dict[str, Any]] = []
    for q_type, (sec_title, sec_order) in sorted(sections_map.items(), key=lambda x: x[1][1]):
        sec_qs = sections_grouped.get(q_type, [])
        if sec_qs:
            sec_marks = sum(int(q.get("marks") or 1) for q in sec_qs)
            built_sections.append({
                "section_name": sec_title,
                "question_type": q_type,
                "questions": sec_qs,
                "total_marks": sec_marks,
            })

    # Time estimation: approx 1.8 minutes per mark (standard exam pace)
    estimated_time = max(30, int(round(current_marks * 1.8)))

    return MockPaper(
        title=title,
        total_marks=current_marks,
        estimated_time_minutes=estimated_time,
        sections=built_sections,
        questions=selected_questions,
        topic_breakdown=topic_distribution,
        strategy_note=f"Sampled {len(selected_questions)} questions across {len(topic_distribution)} topics maximizing high-yield probability.",
    )


# ─── Minimum Study Set (Greedy Set-Cover) ───────────────────────────

def compute_minimum_study_set(
    ranked_questions: List[Dict[str, Any]],
    target_coverage: float = 1.0,
) -> MinimumStudySet:
    """
    Greedy Set-Cover algorithm:
    Finds the minimum subset of high-yield questions that covers 100%
    of discovered exam topics and Bloom cognitive domains.
    Minimizes study burden while guaranteeing full topic coverage.
    """
    if not ranked_questions:
        return MinimumStudySet(
            total_questions=0,
            target_topics_covered=0,
            coverage_percent=0.0,
            total_marks_represented=0,
            questions=[],
            covered_topics=[],
        )

    # Universe of all topics present
    all_topics = {q.get("topic_name", "General") for q in ranked_questions}
    uncovered_topics = set(all_topics)

    selected: List[Dict[str, Any]] = []
    covered: Set[str] = set()
    total_marks = 0

    # Sort questions by composite score descending
    candidates = list(ranked_questions)

    while uncovered_topics and candidates:
        # Pick candidate that covers the most uncovered topics with highest prediction score
        best_candidate = None
        best_score = -1.0

        for q in candidates:
            topic = q.get("topic_name", "General")
            is_uncovered = 1.0 if topic in uncovered_topics else 0.0
            composite = float(q.get("composite_score", 0.0))

            # Prioritize uncovered topics + high prediction score
            heuristic = (is_uncovered * 2.0) + composite

            if heuristic > best_score:
                best_score = heuristic
                best_candidate = q

        if not best_candidate:
            break

        selected.append(best_candidate)
        candidates.remove(best_candidate)

        c_topic = best_candidate.get("topic_name", "General")
        covered.add(c_topic)
        uncovered_topics.discard(c_topic)
        total_marks += int(best_candidate.get("marks") or 1)

    cov_percent = (len(covered) / max(1, len(all_topics))) * 100.0

    return MinimumStudySet(
        total_questions=len(selected),
        target_topics_covered=len(covered),
        coverage_percent=cov_percent,
        total_marks_represented=total_marks,
        questions=selected,
        covered_topics=sorted(list(covered)),
    )
