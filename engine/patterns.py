"""
Exam-Hacking Engine — Pattern Analysis
Analyzes 7 core statistical exam patterns across the question corpus:
1. Topic Frequency: Ranks topics by frequency and marks weightage
2. Year-over-Year Trends: Time-series counts and trajectories per topic
3. Option-Position Bias: Chi-squared test on A/B/C/D option distribution
4. Distractor Analysis: Analyzes wrong options, deceptive templates, and length bias
5. Question-Setter Habits: K-Means clustering on TF-IDF stylistic features
6. Concept Co-occurrence: Identifies topic pairs that frequently appear in the same exam
7. Difficulty Estimation: Composite metric based on length, vocabulary rarity, math, and Bloom's tier
"""

import collections
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from engine import bloom_classifier, db, topic_model

try:
    from scipy.stats import chisquare
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ─── 1. Topic Frequency Analysis ──────────────────────────────────

def analyze_topic_frequency(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ranks topics by question count and total marks weightage.
    """
    if not questions:
        return []

    # Run topic modeling to categorize questions
    tm_result = topic_model.run_topic_modeling(questions)
    topic_map = {t.topic_id: t for t in tm_result.topics}

    freq_data: Dict[int, Dict[str, Any]] = {}
    total_marks = 0

    for a in tm_result.assignments:
        q = next((item for item in questions if item["id"] == a.question_id), None)
        marks = int(q.get("marks") or 1) if q else 1
        total_marks += marks

        if a.topic_id not in freq_data:
            t_def = topic_map.get(a.topic_id)
            freq_data[a.topic_id] = {
                "topic_id": a.topic_id,
                "name": a.topic_name,
                "keywords": t_def.top_keywords if t_def else [],
                "question_count": 0,
                "total_marks": 0,
                "weightage_percent": 0.0,
            }

        freq_data[a.topic_id]["question_count"] += 1
        freq_data[a.topic_id]["total_marks"] += marks

    results = list(freq_data.values())
    for r in results:
        r["weightage_percent"] = round((r["total_marks"] / total_marks) * 100.0, 1) if total_marks > 0 else 0.0

    results.sort(key=lambda x: x["total_marks"], reverse=True)
    return results


# ─── 2. Year-over-Year Trends ─────────────────────────────────────

def analyze_yoy_trends(
    questions: List[Dict[str, Any]],
    documents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Computes time-series appearance of topics across examination years.
    """
    doc_year_map = {d["id"]: d.get("year") for d in documents}
    tm_result = topic_model.run_topic_modeling(questions)
    topic_map = {t.topic_id: t.name for t in tm_result.topics}

    years = sorted(list({y for y in doc_year_map.values() if y is not None}))
    if not years:
        return {"years": [], "trends": []}

    # topic_id -> {year -> count}
    matrix: Dict[int, Dict[int, int]] = {t_id: {y: 0 for y in years} for t_id in topic_map}

    for a in tm_result.assignments:
        q = next((item for item in questions if item["id"] == a.question_id), None)
        if q:
            doc_year = doc_year_map.get(q["document_id"])
            if doc_year in matrix[a.topic_id]:
                matrix[a.topic_id][doc_year] += 1

    trends = []
    for t_id, year_counts in matrix.items():
        counts_list = [year_counts[y] for y in years]
        # Calculate trajectory (rising, steady, declining)
        trajectory = "steady"
        if len(counts_list) >= 2:
            diff = counts_list[-1] - counts_list[0]
            if diff > 0:
                trajectory = "rising"
            elif diff < 0:
                trajectory = "declining"

        trends.append({
            "topic_id": t_id,
            "topic_name": topic_map[t_id],
            "counts_by_year": year_counts,
            "trajectory": trajectory,
            "total": sum(counts_list),
        })

    trends.sort(key=lambda x: x["total"], reverse=True)
    return {"years": years, "trends": trends}


# ─── 3. Option-Position Bias (Chi-Squared) ─────────────────────────

def analyze_option_bias(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyzes MCQ correct answer distribution (A, B, C, D) using Pearson's Chi-squared test.
    Determines if question setters favor specific positions (e.g. B or C).
    """
    mcqs = [q for q in questions if q.get("question_type") == "mcq" and q.get("answer")]
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}

    for q in mcqs:
        ans_raw = q["answer"].strip().upper()
        # Clean answer to option letter
        ans_match = re.search(r"\b([A-D])\b", ans_raw)
        if ans_match:
            counts[ans_match.group(1)] += 1

    total = sum(counts.values())
    if total == 0:
        return {
            "total_mcqs": 0,
            "counts": counts,
            "percentages": {k: 0.0 for k in counts},
            "chi2_statistic": 0.0,
            "p_value": 1.0,
            "has_significant_bias": False,
            "favored_option": None,
            "insight": "No answered MCQs found in the corpus.",
        }

    percentages = {k: round((v / total) * 100.0, 1) for k, v in counts.items()}
    favored_opt = max(counts, key=counts.get)

    chi2_stat = 0.0
    p_val = 1.0
    has_bias = False

    if HAS_SCIPY and total >= 4:
        observed = [counts["A"], counts["B"], counts["C"], counts["D"]]
        expected = [total / 4.0] * 4
        res = chisquare(observed, expected)
        chi2_stat = round(float(res.statistic), 3)
        p_val = round(float(res.pvalue), 4)
        has_bias = p_val < 0.05

    if has_bias:
        insight = f"Statistically significant bias detected (p = {p_val}). Option '{favored_opt}' is favored ({percentages[favored_opt]}% of answers)."
    elif total < 10:
        insight = f"Sample size ({total} MCQs) too small for statistical certainty. Option '{favored_opt}' leads at {percentages[favored_opt]}%."
    else:
        insight = f"Option distribution is balanced (p = {p_val}). No significant setter bias detected."

    return {
        "total_mcqs": total,
        "counts": counts,
        "percentages": percentages,
        "chi2_statistic": chi2_stat,
        "p_value": p_val,
        "has_significant_bias": has_bias,
        "favored_option": favored_opt,
        "insight": insight,
    }


# ─── 4. Distractor Analysis ───────────────────────────────────────

def analyze_distractors(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Examines incorrect MCQ options to detect distractor patterns:
    - Longest option bias (is the longest option disproportionately the correct answer?)
    - Numerical opposite traps (+x vs -x)
    """
    import json
    mcqs = [q for q in questions if q.get("question_type") == "mcq" and q.get("options_json")]
    longest_is_correct = 0
    analyzed_count = 0
    opposite_traps = 0

    for q in mcqs:
        options = json.loads(q["options_json"]) if isinstance(q["options_json"], str) else q.get("options", [])
        ans_raw = (q.get("answer") or "").strip().upper()
        ans_match = re.search(r"\b([A-D])\b", ans_raw)
        if not ans_match or len(options) < 2:
            continue

        correct_label = ans_match.group(1)
        analyzed_count += 1

        # Check longest option
        lengths = {opt.get("label"): len(opt.get("text", "").strip()) for opt in options}
        longest_label = max(lengths, key=lengths.get)
        if longest_label == correct_label and lengths[longest_label] > 0:
            longest_is_correct += 1

        # Check numerical sign traps (e.g. +5 and -5)
        opt_texts = [opt.get("text", "") for opt in options]
        has_pos_neg = any(
            re.search(r"^\+", t) and any(re.search(r"^\-", o) for o in opt_texts) for t in opt_texts
        )
        if has_pos_neg:
            opposite_traps += 1

    longest_pct = round((longest_is_correct / analyzed_count) * 100.0, 1) if analyzed_count > 0 else 0.0

    return {
        "analyzed_mcqs": analyzed_count,
        "longest_option_correct_count": longest_is_correct,
        "longest_option_correct_percent": longest_pct,
        "opposite_traps_count": opposite_traps,
        "longest_bias_detected": longest_pct >= 40.0,
        "insight": (
            f"In {longest_pct}% of questions, the longest option is the correct answer. "
            f"{'Beware of verbose correct options.' if longest_pct >= 40 else 'Option lengths are balanced.'}"
        ),
    }


# ─── 5. Question-Setter Habits (Style Clustering) ────────────────

def analyze_setter_habits(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clusters questions into 3 setter archetypes via K-Means on TF-IDF vectors:
    - Derivation / Theory Heavy
    - Numerical / Formula Application
    - Conceptual / Definition Focused
    """
    total = len(questions)
    if total < 3 or not HAS_SKLEARN:
        return []

    texts = [q.get("question_text", "") for q in questions]

    try:
        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        n_clusters = min(3, total)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(tfidf)

        clusters = []
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
            q_in_cluster = [questions[i] for i in cluster_indices]

            # Top keywords in cluster center
            center = kmeans.cluster_centers_[cluster_id]
            top_words_idx = center.argsort()[-4:][::-1]
            top_words = [feature_names[idx] for idx in top_words_idx if center[idx] > 0]

            # Determine dominant archetype
            num_math = sum(1 for q in q_in_cluster if q.get("has_math") or q.get("question_type") == "numerical")
            num_long = sum(1 for q in q_in_cluster if q.get("question_type") == "long_answer")

            if num_math > len(q_in_cluster) / 2:
                style_name = "Quantitative & Computational"
            elif num_long > len(q_in_cluster) / 2:
                style_name = "Theoretical & Derivational"
            else:
                style_name = "Foundational & Direct Recall"

            clusters.append({
                "cluster_id": cluster_id,
                "style_name": style_name,
                "keywords": top_words,
                "question_count": len(q_in_cluster),
                "percentage": round((len(q_in_cluster) / total) * 100.0, 1),
            })

        clusters.sort(key=lambda x: x["question_count"], reverse=True)
        return clusters

    except Exception as e:
        print(f"[SetterHabits] Clustering error: {e}")
        return []


# ─── 6. Concept Co-occurrence ────────────────────────────────────

def analyze_concept_cooccurrence(
    questions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Computes which topic pairs frequently appear together in the same document.
    """
    if len(questions) < 4:
        return []

    # Map each question to a topic
    tm_result = topic_model.run_topic_modeling(questions)
    topic_map = {t.topic_id: t.name for t in tm_result.topics}
    q_topic_map = {a.question_id: a.topic_id for a in tm_result.assignments}

    # Group topics by document_id
    doc_topics: Dict[int, set] = collections.defaultdict(set)
    for q in questions:
        t_id = q_topic_map.get(q["id"])
        if t_id is not None:
            doc_topics[q["document_id"]].add(t_id)

    # Count co-occurrences
    pair_counts = collections.defaultdict(int)
    for t_set in doc_topics.values():
        t_list = sorted(list(t_set))
        for i in range(len(t_list)):
            for j in range(i + 1, len(t_list)):
                pair_counts[(t_list[i], t_list[j])] += 1

    results = []
    total_docs = len(doc_topics)
    for (t1, t2), count in pair_counts.items():
        if count >= 1 and t1 in topic_map and t2 in topic_map:
            results.append({
                "topic_1": topic_map[t1],
                "topic_2": topic_map[t2],
                "cooccurrence_count": count,
                "paper_frequency_percent": round((count / total_docs) * 100.0, 1) if total_docs > 0 else 0.0,
            })

    results.sort(key=lambda x: x["cooccurrence_count"], reverse=True)
    return results[:10]


# ─── 7. Difficulty Estimation ────────────────────────────────────

def compute_question_difficulty(q: Dict[str, Any]) -> Tuple[int, str]:
    """
    Calculates composite difficulty score (0 - 100):
    - Length complexity (0 - 25 pts)
    - Mathematics presence (0 - 25 pts)
    - Bloom's cognitive level (0 - 35 pts)
    - Marks allotment (0 - 15 pts)
    Returns (score, difficulty_tier: 'Easy', 'Medium', 'Hard').
    """
    stem = q.get("question_text", "")
    words = stem.split()

    # 1. Length factor
    length_score = min(25, int(len(words) * 0.8))

    # 2. Math factor
    math_score = 25 if q.get("has_math") or q.get("question_type") == "numerical" else 0

    # 3. Bloom's Taxonomy factor
    bloom_res = bloom_classifier.classify_bloom_level(
        stem,
        question_type=q.get("question_type"),
        has_math=bool(q.get("has_math")),
    )
    # Tier ranges from 1 (remember) to 6 (create)
    bloom_score = int((bloom_res.tier / 6.0) * 35)

    # 4. Marks factor
    marks = int(q.get("marks") or 1)
    marks_score = min(15, marks * 3)

    raw_score = length_score + math_score + bloom_score + marks_score
    score = max(5, min(100, raw_score))

    if score < 40:
        tier = "Easy"
    elif score < 70:
        tier = "Medium"
    else:
        tier = "Hard"

    return score, tier


def analyze_difficulty_distribution(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes corpus-wide difficulty distribution across Easy, Medium, and Hard tiers.
    """
    total = len(questions)
    if total == 0:
        return {
            "total": 0,
            "counts": {"Easy": 0, "Medium": 0, "Hard": 0},
            "percentages": {"Easy": 0.0, "Medium": 0.0, "Hard": 0.0},
            "avg_difficulty": 0.0,
        }

    counts = {"Easy": 0, "Medium": 0, "Hard": 0}
    scores = []

    for q in questions:
        score, tier = compute_question_difficulty(q)
        counts[tier] += 1
        scores.append(score)

    avg_score = round(float(np.mean(scores)), 1)
    percentages = {tier: round((count / total) * 100.0, 1) for tier, count in counts.items()}

    return {
        "total": total,
        "counts": counts,
        "percentages": percentages,
        "avg_difficulty": avg_score,
    }


# ─── Master Pattern Report ────────────────────────────────────────

def generate_pattern_report(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates complete 7-pattern exam intelligence report from database.
    """
    questions = db.list_questions(valid_only=True, db_path=db_path)
    documents = db.list_documents(db_path=db_path)

    topic_freq = analyze_topic_frequency(questions)
    yoy_trends = analyze_yoy_trends(questions, documents)
    option_bias = analyze_option_bias(questions)
    distractor_data = analyze_distractors(questions)
    setter_habits = analyze_setter_habits(questions)
    cooccurrence = analyze_concept_cooccurrence(questions)
    difficulty = analyze_difficulty_distribution(questions)
    bloom_dist = bloom_classifier.compute_bloom_distribution(questions)

    return {
        "total_questions": len(questions),
        "total_documents": len(documents),
        "topic_frequency": topic_freq,
        "yoy_trends": yoy_trends,
        "option_bias": option_bias,
        "distractors": distractor_data,
        "setter_habits": setter_habits,
        "cooccurrence": cooccurrence,
        "difficulty": difficulty,
        "bloom_distribution": bloom_dist,
    }
