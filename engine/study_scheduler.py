"""
Exam-Hacking Engine — Study Scheduler (Phase 6)
Implements SM-2 Spaced Repetition and Intelligent Day-by-Day Study Planning.
Features:
- SM-2 Spaced Repetition algorithm adapted for exam preparation
- Prioritization of high-probability, high-weightage, and weak questions first
- Dynamic time budgeting per question based on type, marks, and difficulty
- Day-by-day study calendar from current date to exam day
- Automatic review session scheduling at 1, 3, 7, 14, 30 day intervals
- Interactive progress tracking ("not_started", "needs_review", "confident", "mastered")
- Readiness score calculation
"""

from datetime import date, datetime, timedelta
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import config
from engine import db


@dataclass
class SM2Record:
    status: str            # 'not_started', 'needs_review', 'confident', 'mastered'
    easiness_factor: float # EF (1.3 to 2.5+)
    interval_days: int     # Days until next repetition
    repetition_count: int  # Number of successful consecutive reviews
    quality_score: int     # Last rating (0-5)
    next_review_date: str  # YYYY-MM-DD
    last_reviewed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "easiness_factor": round(self.easiness_factor, 2),
            "interval_days": self.interval_days,
            "repetition_count": self.repetition_count,
            "quality_score": self.quality_score,
            "next_review_date": self.next_review_date,
            "last_reviewed_at": self.last_reviewed_at,
        }


@dataclass
class ScheduledTask:
    question_id: int
    question_number: str
    question_text: str
    question_type: str
    marks: int
    topic_name: str
    difficulty_tier: str
    task_type: str         # 'learn' or 'review'
    estimated_minutes: int
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_number": self.question_number,
            "question_text": self.question_text,
            "question_type": self.question_type,
            "marks": self.marks,
            "topic_name": self.topic_name,
            "difficulty_tier": self.difficulty_tier,
            "task_type": self.task_type,
            "estimated_minutes": self.estimated_minutes,
            "status": self.status,
        }


@dataclass
class DaySchedule:
    day_number: int
    date_str: str          # YYYY-MM-DD
    target_minutes: int
    allocated_minutes: int
    focus_topics: List[str]
    tasks: List[ScheduledTask]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day_number": self.day_number,
            "date_str": self.date_str,
            "target_minutes": self.target_minutes,
            "allocated_minutes": self.allocated_minutes,
            "focus_topics": self.focus_topics,
            "tasks": [t.to_dict() for t in self.tasks],
        }


@dataclass
class StudyPlan:
    target_exam_date: str
    days_remaining: int
    daily_study_hours: float
    total_questions: int
    total_study_minutes_estimated: int
    readiness_score_percent: float
    progress_summary: Dict[str, int]
    daily_schedule: List[DaySchedule]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_exam_date": self.target_exam_date,
            "days_remaining": self.days_remaining,
            "daily_study_hours": self.daily_study_hours,
            "total_questions": self.total_questions,
            "total_study_minutes_estimated": self.total_study_minutes_estimated,
            "readiness_score_percent": round(self.readiness_score_percent, 1),
            "progress_summary": self.progress_summary,
            "daily_schedule": [d.to_dict() for d in self.daily_schedule],
        }


# ─── SM-2 Spaced Repetition Core ─────────────────────────────────────

def compute_sm2_update(
    rating: int,
    current_ef: float = 2.5,
    current_interval: int = 1,
    repetition_count: int = 0,
    base_date: Optional[date] = None,
) -> SM2Record:
    """
    Computes SuperMemo-2 (SM-2) interval and easiness factor updates:
    - rating: 0 (blackout) to 5 (perfect recall)
    - EF' = max(1.3, EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)))
    - Intervals: I(1)=1 day, I(2)=3 days, I(n)=round(I(n-1) * EF)
    """
    q = max(0, min(5, int(rating)))
    today_dt = base_date or date.today()

    # Calculate new Easiness Factor
    new_ef = current_ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    new_ef = max(1.30, min(3.20, new_ef))

    if q < 3:
        # Failed recall -> reset repetitions and review tomorrow
        new_repetitions = 0
        new_interval = 1
        status = "needs_review"
    else:
        # Successful recall
        if repetition_count == 0:
            new_interval = 1
        elif repetition_count == 1:
            new_interval = 3
        else:
            new_interval = int(round(current_interval * new_ef))
            new_interval = max(new_interval, current_interval + 1)

        new_repetitions = repetition_count + 1
        status = "mastered" if (new_repetitions >= 3 and q == 5) else "confident"

    next_review_dt = today_dt + timedelta(days=new_interval)

    return SM2Record(
        status=status,
        easiness_factor=new_ef,
        interval_days=new_interval,
        repetition_count=new_repetitions,
        quality_score=q,
        next_review_date=next_review_dt.isoformat(),
        last_reviewed_at=datetime.now().isoformat(),
    )


# ─── Time Estimation per Question ───────────────────────────────────

def estimate_question_study_time_minutes(question: Dict[str, Any]) -> int:
    """
    Estimates study time in minutes for a question based on its format, marks, and tier.
    MCQ: 4 min, Short Answer: 8 min, Long Answer: 18 min, Numerical: 14 min.
    Adjusted by difficulty tier and marks weightage.
    """
    q_type = question.get("question_type", "short_answer")
    marks = int(question.get("marks") or 1)
    diff_tier = question.get("difficulty_tier", "Medium").lower()

    base_times = {
        "mcq": 4,
        "short_answer": 8,
        "long_answer": 18,
        "numerical": 14,
    }
    base = base_times.get(q_type, 8)

    # Adjust for marks
    if marks > 5:
        base += int(marks * 1.5)
    elif marks > 2:
        base += int(marks * 1.0)

    # Adjust for difficulty tier
    tier_multipliers = {
        "easy": 0.85,
        "medium": 1.0,
        "hard": 1.35,
    }
    multiplier = tier_multipliers.get(diff_tier, 1.0)

    return max(3, int(round(base * multiplier)))


# ─── Day-by-Day Schedule Generator ───────────────────────────────────

def generate_study_schedule(
    questions: List[Dict[str, Any]],
    target_exam_date: str,
    daily_study_hours: float = 4.0,
    start_date: Optional[str] = None,
    study_progress_map: Optional[Dict[int, Dict[str, Any]]] = None,
) -> StudyPlan:
    """
    Generates a realistic, day-by-day spaced repetition study schedule:
    - Prioritizes high-yield (composite prediction score), weak/unstudied questions
    - Spaces out new topics in the first 65% of available days
    - Systematically schedules spaced reviews (1, 3, 7, 14 days) in the remaining days
    - Respects daily study time limits
    """
    # Parse dates
    try:
        exam_dt = datetime.strptime(target_exam_date.strip(), "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        # Default to 14 days from now if invalid or missing
        exam_dt = date.today() + timedelta(days=14)

    if start_date:
        try:
            start_dt = datetime.strptime(start_date.strip(), "%Y-%m-%d").date()
        except (ValueError, AttributeError):
            start_dt = date.today()
    else:
        start_dt = date.today()

    days_available = max(1, (exam_dt - start_dt).days)
    daily_budget_minutes = int(daily_study_hours * 60)
    progress_map = study_progress_map or {}

    if not questions:
        return StudyPlan(
            target_exam_date=exam_dt.isoformat(),
            days_remaining=days_available,
            daily_study_hours=daily_study_hours,
            total_questions=0,
            total_study_minutes_estimated=0,
            readiness_score_percent=0.0,
            progress_summary={"not_started": 0, "needs_review": 0, "confident": 0, "mastered": 0},
            daily_schedule=[],
        )

    # Calculate status counts and readiness
    status_counts = {"not_started": 0, "needs_review": 0, "confident": 0, "mastered": 0}
    enriched_questions = []

    for q in questions:
        qid = q["id"]
        prog = progress_map.get(qid)
        st = prog.get("status", "not_started") if prog else "not_started"
        status_counts[st] = status_counts.get(st, 0) + 1

        est_min = estimate_question_study_time_minutes(q)
        composite = float(q.get("composite_score", 0.5))

        enriched_questions.append({
            **q,
            "status": st,
            "estimated_minutes": est_min,
            "composite_score": composite,
            "easiness_factor": float(prog.get("easiness_factor", 2.5)) if prog else 2.5,
            "interval_days": int(prog.get("interval_days", 1)) if prog else 1,
            "repetition_count": int(prog.get("repetition_count", 0)) if prog else 0,
        })

    total_q = len(enriched_questions)
    # Readiness formula: (confident + 1.2 * mastered) / total
    readiness = min(
        100.0,
        ((status_counts["confident"] + (1.2 * status_counts["mastered"])) / max(1, total_q)) * 100.0,
    )

    # Sort questions for learning sequence:
    # 1. Needs review first
    # 2. Not started (highest composite score first)
    # 3. Confident
    # 4. Mastered
    def sort_key(item: Dict[str, Any]):
        priority_map = {"needs_review": 0, "not_started": 1, "confident": 2, "mastered": 3}
        return (priority_map.get(item["status"], 1), -item["composite_score"])

    sorted_for_study = sorted(enriched_questions, key=sort_key)

    # Allocate questions across days
    # Reserve first 70% of days for learning and early reviews, last 30% for consolidation reviews
    learning_days_count = max(1, int(days_available * 0.70))
    daily_schedule: List[DaySchedule] = []

    # Map to track when each question was first studied: question_id -> day_index
    study_day_map: Dict[int, int] = {}
    # Track scheduled reviews: day_index -> list of questions
    future_reviews: Dict[int, List[Dict[str, Any]]] = {d: [] for d in range(days_available)}

    q_idx = 0
    total_plan_minutes = 0

    for d_idx in range(days_available):
        current_day_dt = start_dt + timedelta(days=d_idx)
        current_day_str = current_day_dt.isoformat()
        day_tasks: List[ScheduledTask] = []
        allocated_minutes = 0
        focus_topics_set: Set[str] = set()

        # 1. Process scheduled spaced reviews for this day first
        reviews_today = future_reviews.get(d_idx, [])
        for rq in reviews_today:
            rev_time = max(3, int(rq["estimated_minutes"] * 0.5))  # Review takes ~50% of initial study time
            if allocated_minutes + rev_time <= daily_budget_minutes:
                allocated_minutes += rev_time
                focus_topics_set.add(rq.get("topic_name", "General"))
                day_tasks.append(
                    ScheduledTask(
                        question_id=rq["id"],
                        question_number=str(rq.get("question_number") or ""),
                        question_text=rq.get("question_text", ""),
                        question_type=rq.get("question_type", "short_answer"),
                        marks=int(rq.get("marks") or 1),
                        topic_name=rq.get("topic_name", "General"),
                        difficulty_tier=rq.get("difficulty_tier", "Medium"),
                        task_type="review",
                        estimated_minutes=rev_time,
                        status=rq["status"],
                    )
                )

        # 2. Add new learning questions if within learning window and time remains
        if d_idx < learning_days_count or q_idx < total_q:
            while q_idx < total_q and (allocated_minutes + sorted_for_study[q_idx]["estimated_minutes"]) <= daily_budget_minutes:
                curr_q = sorted_for_study[q_idx]
                q_time = curr_q["estimated_minutes"]
                allocated_minutes += q_time
                focus_topics_set.add(curr_q.get("topic_name", "General"))

                day_tasks.append(
                    ScheduledTask(
                        question_id=curr_q["id"],
                        question_number=str(curr_q.get("question_number") or ""),
                        question_text=curr_q.get("question_text", ""),
                        question_type=curr_q.get("question_type", "short_answer"),
                        marks=int(curr_q.get("marks") or 1),
                        topic_name=curr_q.get("topic_name", "General"),
                        difficulty_tier=curr_q.get("difficulty_tier", "Medium"),
                        task_type="learn",
                        estimated_minutes=q_time,
                        status=curr_q["status"],
                    )
                )
                study_day_map[curr_q["id"]] = d_idx

                # Schedule spaced reviews at +1, +3, +7 days if within exam horizon
                for interval in config.SPACED_REPETITION_INTERVALS[:3]:
                    target_rev_day = d_idx + interval
                    if target_rev_day < days_available:
                        future_reviews.setdefault(target_rev_day, []).append(curr_q)

                q_idx += 1

        total_plan_minutes += allocated_minutes

        daily_schedule.append(
            DaySchedule(
                day_number=d_idx + 1,
                date_str=current_day_str,
                target_minutes=daily_budget_minutes,
                allocated_minutes=allocated_minutes,
                focus_topics=sorted(list(focus_topics_set)),
                tasks=day_tasks,
            )
        )

    # In case any remaining questions couldn't fit tightly, distribute into remaining days
    while q_idx < total_q:
        # Find day with least allocated minutes
        min_day = min(daily_schedule, key=lambda d: d.allocated_minutes)
        curr_q = sorted_for_study[q_idx]
        q_time = curr_q["estimated_minutes"]
        min_day.allocated_minutes += q_time
        total_plan_minutes += q_time
        if curr_q.get("topic_name") and curr_q["topic_name"] not in min_day.focus_topics:
            min_day.focus_topics.append(curr_q["topic_name"])

        min_day.tasks.append(
            ScheduledTask(
                question_id=curr_q["id"],
                question_number=str(curr_q.get("question_number") or ""),
                question_text=curr_q.get("question_text", ""),
                question_type=curr_q.get("question_type", "short_answer"),
                marks=int(curr_q.get("marks") or 1),
                topic_name=curr_q.get("topic_name", "General"),
                difficulty_tier=curr_q.get("difficulty_tier", "Medium"),
                task_type="learn",
                estimated_minutes=q_time,
                status=curr_q["status"],
            )
        )
        q_idx += 1

    return StudyPlan(
        target_exam_date=exam_dt.isoformat(),
        days_remaining=days_available,
        daily_study_hours=daily_study_hours,
        total_questions=total_q,
        total_study_minutes_estimated=total_plan_minutes,
        readiness_score_percent=readiness,
        progress_summary=status_counts,
        daily_schedule=daily_schedule,
    )


# ─── Interactive Progress Updater ────────────────────────────────────

def record_question_review(
    question_id: int,
    rating: int,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Updates the SM-2 progress record for a question after a study session:
    - rating: 0 to 5
    - Persists updated status, EF, interval, and next review date to SQLite
    """
    current_record = db.get_study_progress_for_question(question_id, db_path=db_path)
    if current_record:
        current_ef = float(current_record.get("easiness_factor", 2.5))
        current_interval = int(current_record.get("interval_days", 1))
        repetition_count = int(current_record.get("repetition_count", 0))
    else:
        current_ef = 2.5
        current_interval = 1
        repetition_count = 0

    sm2_update = compute_sm2_update(
        rating=rating,
        current_ef=current_ef,
        current_interval=current_interval,
        repetition_count=repetition_count,
    )

    db.save_study_progress(
        question_id=question_id,
        status=sm2_update.status,
        easiness_factor=sm2_update.easiness_factor,
        interval_days=sm2_update.interval_days,
        repetition_count=sm2_update.repetition_count,
        quality_score=sm2_update.quality_score,
        next_review_date=sm2_update.next_review_date,
        db_path=db_path,
    )

    return sm2_update.to_dict()
