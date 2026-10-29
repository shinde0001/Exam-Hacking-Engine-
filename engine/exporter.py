"""
Exam-Hacking Engine — Exporter (Phase 7)
Generates downloadable artifacts:
1. Publication-ready PDF Mock Exam Papers (via ReportLab)
2. Anki Flashcard Decks (.txt / .tsv ready for Anki import)
3. CSV Exports for Question Banks & Predictions
4. CSV Exports for Spaced Repetition Study Plans
"""

import csv
import io
import os
from typing import Any, Dict, List

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


# ─── 1. CSV Exports ──────────────────────────────────────────────────

def export_questions_to_csv(questions: List[Dict[str, Any]]) -> str:
    """
    Export question list and prediction metrics to a CSV string.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "Question ID",
        "Number",
        "Type",
        "Marks",
        "Topic",
        "Bloom Level",
        "Difficulty Tier",
        "Composite Score",
        "Confidence Badge",
        "Confidence Label",
        "Question Text",
        "Options",
        "Answer",
    ])

    for q in questions:
        options = q.get("options") or []
        opts_formatted = " | ".join(f"{opt.get('label', '')}: {opt.get('text', '')}" for opt in options)
        writer.writerow([
            q.get("id", ""),
            q.get("question_number", ""),
            q.get("question_type", ""),
            q.get("marks", 1),
            q.get("topic_name", "General"),
            q.get("bloom_level", ""),
            q.get("difficulty_tier", ""),
            q.get("composite_score", ""),
            q.get("confidence_badge", ""),
            q.get("confidence_label", ""),
            q.get("question_text", ""),
            opts_formatted,
            q.get("answer", ""),
        ])

    return output.getvalue()


def export_study_plan_to_csv(daily_schedule: List[Dict[str, Any]]) -> str:
    """
    Export day-by-day study schedule to a CSV string.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Day",
        "Date",
        "Task Type",
        "Estimated Minutes",
        "Question Number",
        "Question Type",
        "Marks",
        "Topic",
        "Status",
        "Question Text",
    ])

    for day in daily_schedule:
        day_num = day.get("day_number", "")
        date_str = day.get("date_str", "")
        tasks = day.get("tasks", [])

        for task in tasks:
            writer.writerow([
                day_num,
                date_str,
                task.get("task_type", ""),
                task.get("estimated_minutes", ""),
                task.get("question_number", ""),
                task.get("question_type", ""),
                task.get("marks", 1),
                task.get("topic_name", "General"),
                task.get("status", ""),
                task.get("question_text", ""),
            ])

    return output.getvalue()


# ─── 2. Anki Flashcards Export ───────────────────────────────────────

def export_questions_to_anki(questions: List[Dict[str, Any]]) -> str:
    """
    Exports questions to Anki tab-delimited flashcard format:
    Front [TAB] Back [TAB] Tags [NEWLINE]
    For MCQs: Front contains question + formatted options; Back contains answer.
    For Subjective: Front contains question + marks; Back contains answer/solution.
    """
    output = io.StringIO()

    for q in questions:
        q_text = (q.get("question_text") or "").strip()
        q_type = q.get("question_type", "short_answer")
        marks = q.get("marks", 1)
        topic = (q.get("topic_name") or "General").replace(" ", "_")
        bloom = (q.get("bloom_level") or "general").replace(" ", "_")
        diff = (q.get("difficulty_tier") or "Medium").replace(" ", "_")

        options = q.get("options") or []
        answer = (q.get("answer") or "").strip()

        # Build Front Card HTML
        front_parts = [f"<p><strong>Question ({marks} Mark{'s' if marks > 1 else ''}):</strong></p>", f"<p>{q_text}</p>"]

        if options:
            front_parts.append("<div style='margin-top: 10px;'>")
            for opt in options:
                lbl = opt.get("label", "")
                txt = opt.get("text", "")
                front_parts.append(f"<div><strong>({lbl})</strong> {txt}</div>")
            front_parts.append("</div>")

        front_html = "".join(front_parts).replace("\t", " ").replace("\n", "<br>")

        # Build Back Card HTML
        back_parts = []
        if answer:
            back_parts.append(f"<p><strong>Correct Answer:</strong> {answer}</p>")
        else:
            back_parts.append("<p><em>(Refer to textbook/notes for detailed solution)</em></p>")

        back_parts.append(f"<p style='color: #666; font-size: 0.9em;'>Topic: {topic} | Bloom: {bloom} | Type: {q_type}</p>")
        back_html = "".join(back_parts).replace("\t", " ").replace("\n", "<br>")

        tags = f"ExamEngine {topic} {bloom} {diff} {q_type}"

        # Write Anki TSV row
        output.write(f"{front_html}\t{back_html}\t{tags}\n")

    return output.getvalue()


# ─── 3. PDF Mock Exam Paper Generation ──────────────────────────────

def clean_text_for_pdf(text: str) -> str:
    """Escape XML characters for ReportLab Paragraph compatibility."""
    if not text:
        return ""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def generate_mock_paper_pdf(
    mock_paper_dict: Dict[str, Any],
    output_filepath: str,
) -> str:
    """
    Generates a printable PDF mock examination paper with instructions,
    clean sections, multiple choice formatting, and marking schemes.
    """
    if not HAS_REPORTLAB:
        raise RuntimeError("ReportLab is required for PDF generation.")

    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    doc = SimpleDocTemplate(
        output_filepath,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ExamTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        alignment=1,  # Centered
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "ExamSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        leading=15,
        alignment=1,  # Centered
        textColor=colors.HexColor("#64748b"),
        spaceAfter=14,
    )

    section_header_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )

    question_style = ParagraphStyle(
        "QuestionText",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=4,
    )

    option_style = ParagraphStyle(
        "OptionText",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        leftIndent=15,
        textColor=colors.HexColor("#334155"),
        spaceAfter=2,
    )

    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Italic"],
        fontSize=9,
        leading=12,
        alignment=2,  # Right aligned
        textColor=colors.HexColor("#64748b"),
        spaceAfter=8,
    )

    story = []

    # Title & Subtitle
    paper_title = clean_text_for_pdf(mock_paper_dict.get("title", "Predicted Mock Examination Paper"))
    story.append(Paragraph(paper_title, title_style))

    total_marks = mock_paper_dict.get("total_marks", 50)
    est_time = mock_paper_dict.get("estimated_time_minutes", 90)
    subtitle = f"<b>Total Marks:</b> {total_marks} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Time Allowed:</b> {est_time} Minutes &nbsp;&nbsp;|&nbsp;&nbsp; <b>Predicted by:</b> Exam-Hacking Engine"
    story.append(Paragraph(subtitle, subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#cbd5e1"), spaceAfter=14))

    # General Instructions Box
    instructions_text = (
        "<b>General Instructions:</b><br/>"
        "1. All questions are compulsory. Marks for each question are indicated in brackets.<br/>"
        "2. Section A contains objective Multiple Choice Questions. Select the single best answer.<br/>"
        "3. Section B & C contain descriptive short and long answer questions. Show all calculations and diagrams where applicable.<br/>"
        "4. Neat handwriting and clear presentation carry merit."
    )
    instr_style = ParagraphStyle(
        "InstrStyle",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#475569"),
    )
    instr_table = Table([[Paragraph(instructions_text, instr_style)]], colWidths=[500])
    instr_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("PADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(instr_table)
    story.append(Spacer(1, 14))

    # Sections & Questions
    sections = mock_paper_dict.get("sections") or []
    global_q_num = 1

    for sec in sections:
        sec_name = clean_text_for_pdf(sec.get("section_name", "Section"))
        sec_marks = sec.get("total_marks", 0)
        story.append(Paragraph(f"<b>{sec_name}</b> ({sec_marks} Marks)", section_header_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceAfter=8))

        sec_questions = sec.get("questions") or []
        for q in sec_questions:
            q_text = clean_text_for_pdf(q.get("question_text", ""))
            marks = q.get("marks", 1)
            topic = clean_text_for_pdf(q.get("topic_name", "General"))

            # Question Paragraph with marks
            q_para = f"<b>Q{global_q_num}.</b> {q_text} &nbsp;&nbsp;<b>[{marks} Mark{'s' if marks > 1 else ''}]</b>"
            story.append(Paragraph(q_para, question_style))

            # Options if MCQ
            options = q.get("options") or []
            if options:
                opt_row_1 = []
                opt_row_2 = []
                for idx, opt in enumerate(options):
                    lbl = clean_text_for_pdf(opt.get("label", ""))
                    txt = clean_text_for_pdf(opt.get("text", ""))
                    opt_html = f"<b>({lbl})</b> {txt}"
                    if idx < 2:
                        opt_row_1.append(Paragraph(opt_html, option_style))
                    else:
                        opt_row_2.append(Paragraph(opt_html, option_style))

                table_rows = []
                if opt_row_1:
                    while len(opt_row_1) < 2:
                        opt_row_1.append(Paragraph("", option_style))
                    table_rows.append(opt_row_1)
                if opt_row_2:
                    while len(opt_row_2) < 2:
                        opt_row_2.append(Paragraph("", option_style))
                    table_rows.append(opt_row_2)

                if table_rows:
                    t = Table(table_rows, colWidths=[240, 240])
                    t.setStyle(TableStyle([
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ]))
                    story.append(t)

            story.append(Paragraph(f"<i>[Topic: {topic}]</i>", meta_style))
            story.append(Spacer(1, 4))
            global_q_num += 1

        story.append(Spacer(1, 8))

    doc.build(story)
    return output_filepath
