"""
Exam-Hacking Engine — Flask Application
Web UI and API for Phase 1: Ingestion, Status Tracking, and Document Management.
"""

import os
import config
from werkzeug.utils import secure_filename
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    Response,
    send_file,
    url_for,
)

from engine import (
    db,
    ingest,
    extractor,
    vectorstore,
    syllabus_mapper,
    patterns,
    predictor,
    study_scheduler,
    exporter,
)

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH


def is_allowed_file(filename: str) -> bool:
    """Validate file extension against allowed configuration."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Dashboard displaying document counts, ingestion statuses, and document list."""
    try:
        documents = db.list_documents()
        stats = db.get_database_stats()
    except Exception as e:
        # If DB not yet initialized or table missing
        documents = []
        stats = {
            "total_documents": 0,
            "ingested_documents": 0,
            "failed_documents": 0,
            "total_pages": 0,
            "ocr_pages": 0,
        }
        flash(f"Database notice: {e}", "warning")

    return render_template("index.html", documents=documents, stats=stats)


@app.route("/upload", methods=["GET"])
def upload_page():
    """Render the document upload page."""
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload_files():
    """Handle document uploads and trigger ingestion pipeline."""
    if "files" not in request.files:
        flash("No file part selected in upload request.", "danger")
        return redirect(request.url)

    uploaded_files = request.files.getlist("files")
    if not uploaded_files or uploaded_files[0].filename == "":
        flash("No files were selected for upload.", "warning")
        return redirect(request.url)

    year_raw = request.form.get("year", "").strip()
    year = int(year_raw) if year_raw.isdigit() else None
    subject = request.form.get("subject", "").strip() or None
    force_reingest = bool(request.form.get("force_reingest"))

    success_count = 0
    skipped_count = 0
    errors = []

    for file in uploaded_files:
        if not file.filename:
            continue

        if not is_allowed_file(file.filename):
            errors.append(f"Skipped '{file.filename}': Unsupported file format.")
            continue

        safe_name = secure_filename(file.filename)
        dest_path = os.path.join(config.UPLOAD_DIR, safe_name)

        # Avoid name collision on disk while maintaining original basename
        base, ext = os.path.splitext(safe_name)
        counter = 1
        while os.path.exists(dest_path):
            dest_path = os.path.join(config.UPLOAD_DIR, f"{base}_{counter}{ext}")
            counter += 1

        file.save(dest_path)

        # Run ingestion
        try:
            result = ingest.ingest_file(
                dest_path,
                year=year,
                subject=subject,
                force_reingest=force_reingest,
            )
            if result.skipped:
                skipped_count += 1
                # Clean up duplicate file from uploads/ if it points to an existing ingested document
                orig_doc = db.get_document_by_id(result.document_id)
                if orig_doc and orig_doc["filepath"] != dest_path and os.path.exists(dest_path):
                    try:
                        os.remove(dest_path)
                    except OSError:
                        pass
            else:
                success_count += 1
        except Exception as e:
            errors.append(f"Failed to ingest '{file.filename}': {e}")

    if success_count > 0:
        flash(f"Successfully processed {success_count} new document(s).", "success")
    if skipped_count > 0:
        flash(f"{skipped_count} document(s) were already ingested and skipped.", "info")
    for err in errors:
        flash(err, "danger")

    return redirect(url_for("index"))


@app.route("/documents/<int:doc_id>")
def view_document(doc_id: int):
    """View details and extracted pages for a single document."""
    document = db.get_document_by_id(doc_id)
    if not document:
        flash(f"Document with ID {doc_id} not found.", "danger")
        return redirect(url_for("index"))

    pages = db.get_document_pages(doc_id)
    return render_template("document_view.html", document=document, pages=pages)


@app.route("/documents/<int:doc_id>/reingest", methods=["POST"])
def reingest_document(doc_id: int):
    """Trigger re-ingestion for an existing document."""
    document = db.get_document_by_id(doc_id)
    if not document:
        flash(f"Document with ID {doc_id} not found.", "danger")
        return redirect(url_for("index"))

    filepath = document["filepath"]
    if not os.path.exists(filepath):
        flash(f"Source file not found on disk at '{filepath}'.", "danger")
        return redirect(url_for("index"))

    try:
        ingest.ingest_file(
            filepath,
            year=document["year"],
            subject=document["subject"],
            force_reingest=True,
        )
        flash(f"Successfully re-ingested '{document['filename']}'.", "success")
    except Exception as e:
        flash(f"Re-ingestion failed: {e}", "danger")

    return redirect(url_for("view_document", doc_id=doc_id))


@app.route("/documents/<int:doc_id>/delete", methods=["POST"])
def delete_doc(doc_id: int):
    """Delete a document record and its physical uploaded file."""
    document = db.get_document_by_id(doc_id)
    if not document:
        flash(f"Document ID {doc_id} not found.", "danger")
        return redirect(url_for("index"))

    # Remove file from disk if present
    filepath = document["filepath"]
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError as e:
            flash(f"Warning: Could not remove file from disk: {e}", "warning")

    db.delete_document(doc_id)
    flash(f"Document '{document['filename']}' deleted successfully.", "success")
    return redirect(url_for("index"))


@app.route("/documents/<int:doc_id>/extract", methods=["POST"])
def extract_document_questions_route(doc_id: int):
    """Trigger question extraction, auto-type detection, and validation for a document."""
    document = db.get_document_by_id(doc_id)
    if not document:
        flash(f"Document ID {doc_id} not found.", "danger")
        return redirect(url_for("index"))

    try:
        extracted = extractor.extract_and_store_document_questions(doc_id)
        valid_count = sum(1 for q in extracted if q["is_valid"])
        flash(
            f"Extracted {len(extracted)} question(s) ({valid_count} valid for prediction pipeline).",
            "success",
        )
    except Exception as e:
        flash(f"Question extraction failed: {e}", "danger")

    return redirect(url_for("view_document_questions", doc_id=doc_id))


@app.route("/documents/<int:doc_id>/questions")
def view_document_questions(doc_id: int):
    """View structured questions extracted from a document."""
    import json
    document = db.get_document_by_id(doc_id)
    if not document:
        flash(f"Document ID {doc_id} not found.", "danger")
        return redirect(url_for("index"))

    raw_questions = db.get_questions_by_document(doc_id)
    parsed_questions = []
    for q in raw_questions:
        options = json.loads(q["options_json"]) if q["options_json"] else []
        issues = json.loads(q["validation_issues"]) if q["validation_issues"] else []
        parsed_questions.append({
            **q,
            "options": options,
            "issues": issues,
        })

    return render_template(
        "questions_view.html",
        document=document,
        questions=parsed_questions,
    )


@app.route("/search")
def semantic_search_page():
    """Render the semantic question search interface."""
    import json
    query = request.args.get("q", "").strip()
    top_k_raw = request.args.get("top_k", "10")
    top_k = int(top_k_raw) if top_k_raw.isdigit() else 10
    min_score_raw = request.args.get("min_score", "0.4")
    try:
        min_score = float(min_score_raw)
    except ValueError:
        min_score = 0.4

    results = []
    if query:
        raw_results = vectorstore.semantic_search_questions(query, top_k=top_k, min_score=min_score)
        for r in raw_results:
            options = json.loads(r["options_json"]) if r["options_json"] else []
            results.append({
                **r,
                "options": options,
            })

    return render_template(
        "search.html",
        query=query,
        results=results,
        top_k=top_k,
        min_score=min_score,
    )


@app.route("/duplicates")
def duplicates_page():
    """Render the duplicates and answer conflict review dashboard."""
    edges = db.list_similarity_edges()
    conflicts = [e for e in edges if e["has_answer_conflict"]]
    stats = db.get_similarity_stats()

    return render_template(
        "duplicates.html",
        edges=edges,
        conflicts=conflicts,
        stats=stats,
    )


@app.route("/duplicates/scan", methods=["POST"])
def scan_duplicates_route():
    """Trigger full-corpus deduplication and conflict scanning."""
    try:
        edges = vectorstore.find_and_record_duplicates()
        conflicts = [e for e in edges if e["has_conflict"]]
        flash(
            f"Corpus scan complete: Discovered {len(edges)} duplicate pair(s) and {len(conflicts)} answer conflict(s).",
            "success" if not conflicts else "warning",
        )
    except Exception as e:
        flash(f"Duplicate scan failed: {e}", "danger")

    return redirect(url_for("duplicates_page"))


@app.route("/patterns")
def patterns_page():
    """Render the exam pattern intelligence dashboard."""
    report = patterns.generate_pattern_report()
    return render_template("patterns.html", report=report)


@app.route("/syllabus", methods=["GET", "POST"])
def syllabus_page():
    """Render the syllabus mapper and compute chapter exam weightages."""
    raw_syllabus = ""
    chapters = []

    if request.method == "POST":
        raw_syllabus = request.form.get("syllabus_text", "").strip()
        if raw_syllabus:
            parsed_chapters = syllabus_mapper.parse_syllabus_text(raw_syllabus)
            questions = db.list_questions(valid_only=True)
            chapters = syllabus_mapper.map_questions_to_syllabus(questions, parsed_chapters)

    return render_template(
        "syllabus.html",
        raw_syllabus=raw_syllabus,
        chapters=chapters,
    )


@app.route("/predictions")
def predictions_page():
    """Render the master prediction engine dashboard (Phase 5)."""
    report = predictor.run_prediction_pipeline()
    return render_template("predictions.html", report=report)


@app.route("/mock-paper", methods=["GET", "POST"])
def mock_paper_page():
    """Render and customize the generated predicted mock examination paper."""
    title = request.form.get("title") or request.args.get("title") or "Predicted Mock Examination Paper"
    target_marks_raw = request.form.get("target_marks") or request.args.get("target_marks") or "50"
    target_qs_raw = request.form.get("target_questions") or request.args.get("target_questions") or "15"

    try:
        target_marks = int(target_marks_raw)
    except ValueError:
        target_marks = 50

    try:
        target_qs = int(target_qs_raw)
    except ValueError:
        target_qs = 15

    report = predictor.run_prediction_pipeline()
    paper = predictor.generate_mock_paper(
        ranked_questions=report.ranked_questions,
        target_marks=target_marks,
        target_questions=target_qs,
        title=title,
    )

    return render_template("mock_paper.html", paper=paper)


@app.route("/study-plan", methods=["GET", "POST"])
def study_plan_page():
    """Render the SM-2 spaced repetition study planner (Phase 6)."""
    from datetime import date, timedelta
    default_exam_date = (date.today() + timedelta(days=14)).isoformat()

    target_date = request.form.get("target_exam_date") or request.args.get("target_exam_date") or default_exam_date
    daily_hours_raw = request.form.get("daily_study_hours") or request.args.get("daily_study_hours") or "4.0"

    try:
        daily_hours = float(daily_hours_raw)
    except ValueError:
        daily_hours = 4.0

    report = predictor.run_prediction_pipeline()
    study_set = predictor.compute_minimum_study_set(report.ranked_questions)
    progress_map = db.get_study_progress()

    plan = study_scheduler.generate_study_schedule(
        questions=study_set.questions,
        target_exam_date=target_date,
        daily_study_hours=daily_hours,
        study_progress_map=progress_map,
    )

    return render_template("study_plan.html", plan=plan)


# ─── Export Endpoints (Phase 7) ───────────────────────────────────

@app.route("/export/questions/csv")
def export_predictions_csv():
    """Download predictions and question bank as a CSV file."""
    report = predictor.run_prediction_pipeline()
    csv_content = exporter.export_questions_to_csv(report.ranked_questions)
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=exam_predictions.csv"},
    )


@app.route("/export/questions/anki")
def export_anki_flashcards():
    """Download questions formatted as Anki flashcard deck (TSV)."""
    report = predictor.run_prediction_pipeline()
    anki_content = exporter.export_questions_to_anki(report.ranked_questions)
    return Response(
        anki_content,
        mimetype="text/tab-separated-values",
        headers={"Content-Disposition": "attachment; filename=exam_anki_flashcards.txt"},
    )


@app.route("/export/mock-paper/pdf")
def export_mock_paper_pdf_route():
    """Generate and download the predicted mock examination paper as an official PDF."""
    target_marks_raw = request.args.get("target_marks", "50")
    target_qs_raw = request.args.get("target_questions", "15")

    try:
        target_marks = int(target_marks_raw)
    except ValueError:
        target_marks = 50

    try:
        target_qs = int(target_qs_raw)
    except ValueError:
        target_qs = 15

    report = predictor.run_prediction_pipeline()
    paper = predictor.generate_mock_paper(
        ranked_questions=report.ranked_questions,
        target_marks=target_marks,
        target_questions=target_qs,
    )

    pdf_filename = f"predicted_mock_paper_{target_marks}marks.pdf"
    pdf_path = os.path.join(config.EXPORT_DIR, pdf_filename)
    exporter.generate_mock_paper_pdf(paper.to_dict(), pdf_path)

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name="predicted_mock_exam_paper.pdf",
        mimetype="application/pdf",
    )


@app.route("/export/study-plan/csv")
def export_study_plan_csv_route():
    """Download day-by-day study schedule as a CSV file."""
    from datetime import date, timedelta
    default_exam_date = (date.today() + timedelta(days=14)).isoformat()
    target_date = request.args.get("exam_date", default_exam_date)
    daily_hours_raw = request.args.get("hours", "4.0")

    try:
        daily_hours = float(daily_hours_raw)
    except ValueError:
        daily_hours = 4.0

    report = predictor.run_prediction_pipeline()
    study_set = predictor.compute_minimum_study_set(report.ranked_questions)
    progress_map = db.get_study_progress()

    plan = study_scheduler.generate_study_schedule(
        questions=study_set.questions,
        target_exam_date=target_date,
        daily_study_hours=daily_hours,
        study_progress_map=progress_map,
    )

    csv_content = exporter.export_study_plan_to_csv(plan.to_dict()["daily_schedule"])
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=study_plan.csv"},
    )


# ─── API Endpoints ──────────────────────────────────────────────

@app.route("/api/health")
def api_health():
    """System health check and diagnostic status."""
    return jsonify({
        "status": "healthy",
        "tesseract_ocr_available": ingest.is_tesseract_available(),
        "pdfplumber_available": ingest.HAS_PDFPLUMBER,
        "pymupdf_available": ingest.HAS_PYMUPDF,
        "python_docx_available": ingest.HAS_PYTHON_DOCX,
        "database_path": config.DATABASE_PATH,
        "upload_dir": config.UPLOAD_DIR,
    })


@app.route("/api/documents")
def api_list_documents():
    """Return JSON list of documents, optionally filtered by ?status="""
    status_filter = request.args.get("status")
    documents = db.list_documents(status=status_filter)
    return jsonify({"count": len(documents), "documents": documents})


@app.route("/api/documents/<int:doc_id>")
def api_get_document(doc_id: int):
    """Return JSON metadata and pages for a specific document."""
    document = db.get_document_by_id(doc_id)
    if not document:
        return jsonify({"error": "Document not found"}), 404
    pages = db.get_document_pages(doc_id)
    return jsonify({"document": document, "pages": pages})


@app.route("/api/documents/<int:doc_id>/questions")
def api_get_document_questions(doc_id: int):
    """Return JSON questions extracted from a specific document."""
    import json
    document = db.get_document_by_id(doc_id)
    if not document:
        return jsonify({"error": "Document not found"}), 404

    valid_only = request.args.get("valid_only", "false").lower() == "true"
    questions = db.get_questions_by_document(doc_id, valid_only=valid_only)
    parsed = []
    for q in questions:
        q_dict = dict(q)
        q_dict["options"] = json.loads(q["options_json"]) if q["options_json"] else []
        q_dict["validation_issues"] = json.loads(q["validation_issues"]) if q["validation_issues"] else []
        parsed.append(q_dict)

    return jsonify({"document_id": doc_id, "count": len(parsed), "questions": parsed})


@app.route("/api/questions")
def api_list_questions():
    """Return filtered questions across the entire corpus."""
    import json
    q_type = request.args.get("type")
    min_quality_raw = request.args.get("min_quality")
    min_quality = int(min_quality_raw) if min_quality_raw and min_quality_raw.isdigit() else None
    valid_only = request.args.get("valid_only", "true").lower() == "true"

    questions = db.list_questions(
        question_type=q_type,
        min_quality=min_quality,
        valid_only=valid_only,
    )
    parsed = []
    for q in questions:
        q_dict = dict(q)
        q_dict["options"] = json.loads(q["options_json"]) if q["options_json"] else []
        q_dict["validation_issues"] = json.loads(q["validation_issues"]) if q["validation_issues"] else []
        parsed.append(q_dict)

    return jsonify({"count": len(parsed), "questions": parsed})


@app.route("/api/search")
def api_semantic_search():
    """Semantic search API returning matching questions for a natural language query."""
    import json
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Query parameter 'q' is required."}), 400

    top_k_raw = request.args.get("top_k", "10")
    top_k = int(top_k_raw) if top_k_raw.isdigit() else 10
    min_score_raw = request.args.get("min_score", "0.4")
    try:
        min_score = float(min_score_raw)
    except ValueError:
        min_score = 0.4

    raw_results = vectorstore.semantic_search_questions(query, top_k=top_k, min_score=min_score)
    parsed = []
    for r in raw_results:
        options = json.loads(r["options_json"]) if r["options_json"] else []
        parsed.append({
            **r,
            "options": options,
        })

    return jsonify({"query": query, "count": len(parsed), "results": parsed})


@app.route("/api/duplicates")
def api_list_duplicates():
    """Return JSON list of duplicate edges and answer conflicts."""
    edges = db.list_similarity_edges()
    stats = db.get_similarity_stats()
    return jsonify({"stats": stats, "count": len(edges), "duplicates": edges})


@app.route("/api/questions/<int:qid>/similar")
def api_question_similar(qid: int):
    """Return similar questions for a specific question ID."""
    similar = db.get_similar_questions_for_id(qid)
    return jsonify({"question_id": qid, "count": len(similar), "similar": similar})


@app.route("/api/patterns")
def api_get_patterns():
    """Return the complete 7-pattern exam intelligence report as JSON."""
    report = patterns.generate_pattern_report()
    return jsonify(report)


@app.route("/api/syllabus", methods=["POST"])
def api_map_syllabus():
    """Map syllabus JSON or text payload to corpus questions and return chapter weightages."""
    data = request.get_json(silent=True) or {}
    syllabus_text = data.get("syllabus_text") or request.form.get("syllabus_text", "")
    if not syllabus_text:
        return jsonify({"error": "syllabus_text is required"}), 400

    parsed_chapters = syllabus_mapper.parse_syllabus_text(syllabus_text)
    questions = db.list_questions(valid_only=True)
    weightages = syllabus_mapper.map_questions_to_syllabus(questions, parsed_chapters)

    output = []
    for w in weightages:
        output.append({
            "chapter_number": w.chapter_number,
            "title": w.title,
            "subtopics": w.subtopics,
            "question_count": w.question_count,
            "total_marks": w.total_marks,
            "weightage_percent": w.weightage_percent,
            "mapped_questions_count": len(w.mapped_questions),
        })

    return jsonify({"count": len(output), "chapters": output})


@app.route("/api/predictions")
def api_predictions():
    """Return JSON prediction report containing ranked questions, top topics, and cyclic alerts."""
    report = predictor.run_prediction_pipeline()
    return jsonify(report.to_dict())


@app.route("/api/predictions/mock-paper", methods=["POST"])
def api_mock_paper():
    """Generate and return a structured predicted mock examination paper as JSON."""
    data = request.get_json(silent=True) or {}
    target_marks = int(data.get("target_marks") or 50)
    target_qs = int(data.get("target_questions") or 15)
    title = data.get("title") or "Predicted Mock Examination Paper"

    report = predictor.run_prediction_pipeline()
    paper = predictor.generate_mock_paper(
        ranked_questions=report.ranked_questions,
        target_marks=target_marks,
        target_questions=target_qs,
        title=title,
    )
    return jsonify(paper.to_dict())


@app.route("/api/predictions/minimum-study-set")
def api_minimum_study_set():
    """Return the greedy set-cover minimum study set as JSON."""
    report = predictor.run_prediction_pipeline()
    study_set = predictor.compute_minimum_study_set(report.ranked_questions)
    return jsonify(study_set.to_dict())


@app.route("/api/study-plan", methods=["GET", "POST"])
def api_study_plan():
    """Generate and return a day-by-day spaced repetition study plan as JSON."""
    from datetime import date, timedelta
    default_exam_date = (date.today() + timedelta(days=14)).isoformat()

    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        target_date = data.get("target_exam_date") or default_exam_date
        daily_hours = float(data.get("daily_study_hours") or 4.0)
    else:
        target_date = request.args.get("target_exam_date") or default_exam_date
        daily_hours_raw = request.args.get("daily_study_hours") or "4.0"
        try:
            daily_hours = float(daily_hours_raw)
        except ValueError:
            daily_hours = 4.0

    report = predictor.run_prediction_pipeline()
    study_set = predictor.compute_minimum_study_set(report.ranked_questions)
    progress_map = db.get_study_progress()

    plan = study_scheduler.generate_study_schedule(
        questions=study_set.questions,
        target_exam_date=target_date,
        daily_study_hours=daily_hours,
        study_progress_map=progress_map,
    )
    return jsonify(plan.to_dict())


@app.route("/api/study-plan/record-review", methods=["POST"])
def api_record_study_review():
    """Record an SM-2 rating (0-5) for a question and update its progress status."""
    data = request.get_json(silent=True) or {}
    question_id_raw = data.get("question_id")
    rating_raw = data.get("rating")

    if question_id_raw is None or rating_raw is None:
        return jsonify({"error": "question_id and rating are required."}), 400

    try:
        qid = int(question_id_raw)
        rating = int(rating_raw)
    except ValueError:
        return jsonify({"error": "Invalid question_id or rating."}), 400

    updated_record = study_scheduler.record_question_review(qid, rating)
    return jsonify(updated_record)


if __name__ == "__main__":
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
