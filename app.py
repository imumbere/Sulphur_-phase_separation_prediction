"""Flask application for predicting formulation phase separation.

Deploy with: gunicorn app:app
This application serves the original trusted gradient boosting model in random_forest_model.pkl.
"""
from __future__ import annotations

import io
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT / "random_forest_model.pkl")))

# The *order* and *spelling* of these column names match the uploaded model's
# feature_names_in_ metadata. The original training data spells Gylcerine this way.
MODEL_COLUMNS = (
    "microwax (g)",
    "Mineral oil vol (ml)",
    "Mix stop temp",
    "Gylcerine wgt (g)",
    "H2O bath temp",
    "run time (mins)",
)

FIELDS = (
    dict(key="microwax", label="Microwax", unit="g", detail="Weight of microwax", icon="layers", nonnegative=True),
    dict(key="mineral_oil", label="Mineral oil volume", unit="mL", detail="Volume of mineral oil", icon="droplet", nonnegative=True),
    dict(key="mix_stop_temp", label="Mix stop temperature", unit="°C", detail="Temperature when mixing stops", icon="thermometer", nonnegative=False),
    dict(key="glycerine_wgt", label="Glycerine weight", unit="g", detail="Weight of glycerine", icon="flask", nonnegative=True),
    dict(key="h2o_bath_temp", label="Water bath temperature", unit="°C", detail="Water bath set temperature", icon="waves", nonnegative=False),
    dict(key="run_time", label="Run time", unit="min", detail="Total processing time", icon="clock", nonnegative=True),
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # Simple form; no file uploads.

model = None
model_status = None
try:
    if MODEL_PATH.is_file():
        model = joblib.load(MODEL_PATH)  # ONLY load your own trusted local model file.
        if not hasattr(model, "predict"):
            raise TypeError("Saved object does not implement predict().")
        model_names = getattr(model, "feature_names_in_", None)
        if model_names is not None and list(model_names) != list(MODEL_COLUMNS):
            raise ValueError(
                "Model feature names/order differ from the expected training columns. "
                "Check MODEL_COLUMNS in app.py against your model's feature_names_in_."
            )
    else:
        model_status = "Prediction model not installed. Add random_forest_model.pkl to the project root."
except Exception:
    app.logger.exception("Could not load prediction model")
    model = None
    model_status = "Could not load the model. Verify the original .pkl file and scikit-learn version."


def parse_form(form):
    """Validate all values once; use the same rules for predict and PDF export."""
    values = {}
    for field in FIELDS:
        key = field["key"]
        raw = form.get(key, "").strip()
        if not raw:
            raise ValueError(f"Please enter {field['label'].lower()}.")
        try:
            number = float(raw)
        except (ValueError, OverflowError):
            raise ValueError(f"{field['label']} must be a valid number.") from None
        if not math.isfinite(number):
            raise ValueError(f"{field['label']} must be a finite number.")
        if field["nonnegative"] and number < 0:
            raise ValueError(f"{field['label']} cannot be negative.")
        values[key] = number
    return values


def predict_outcome(values):
    if model is None:
        raise RuntimeError(model_status or "Prediction model unavailable.")
    row = [values[field["key"]] for field in FIELDS]
    # A DataFrame keeps training column names, including the spelling in model metadata.
    x = pd.DataFrame([row], columns=MODEL_COLUMNS)
    prediction = model.predict(x)[0]
    if prediction not in (0, 1):
        raise RuntimeError("The model returned a class other than 0 or 1; verify your class mapping.")
    separated = int(prediction) == 1
    return {
        "separated": separated,
        "label": "Phase Separated" if separated else "No Phase Separation",
        "summary": (
            "The model predicts phase separation for these formulation settings."
            if separated else
            "The model predicts no phase separation for these formulation settings."
        ),
    }


def show_form(values=None, result=None, error=None, status=200):
    return (
        render_template(
            "index.html",
            fields=FIELDS,
            values=values or {},
            result=result,
            error=error,
            model_available=model is not None,
            model_status=model_status,
        ),
        status,
    )


@app.get("/")
def home():
    return show_form()


@app.post("/predict")
def predict():
    submitted = {f["key"]: request.form.get(f["key"], "") for f in FIELDS}
    try:
        values = parse_form(request.form)
        result = predict_outcome(values)
        return show_form(values=values, result=result)
    except ValueError as exc:
        return show_form(values=submitted, error=str(exc), status=400)
    except RuntimeError as exc:
        return show_form(values=submitted, error=str(exc), status=503)
    except Exception:
        app.logger.exception("Unexpected prediction error")
        return show_form(values=submitted, error="Prediction failed. Please check the server logs.", status=500)


def build_pdf(values, result):
    """Build a single-page PDF in memory, without storing user inputs."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4, pageCompression=1)
    width, height = A4
    ink = colors.HexColor("#142B3E")
    teal = colors.HexColor("#087F80")
    muted = colors.HexColor("#597080")
    pale = colors.HexColor("#F1F8F8")
    paper = colors.HexColor("#FFFFFF")
    warning = colors.HexColor("#9F4C28")

    pdf.setTitle("Phase Separation Prediction Report")
    pdf.setAuthor("Phase Separation Prediction Application")
    pdf.setSubject("Machine-learning prediction with formulation process inputs")

    # Header band and geometric pharmaceutical mark.
    pdf.setFillColor(ink)
    pdf.rect(0, height - 153, width, 153, fill=1, stroke=0)
    pdf.setLineWidth(1.5)
    pdf.setStrokeColor(colors.HexColor("#3B6E7C"))
    cx, cy, rr = width - 95, height - 75, 39
    vertices = []
    import math as _math
    for j in range(6):
        a = _math.radians(60 * j + 30)
        vertices.append((cx + rr * _math.cos(a), cy + rr * _math.sin(a)))
    for j in range(6):
        start = vertices[j]
        end = vertices[(j + 1) % 6]
        pdf.line(start[0], start[1], end[0], end[1])
    pdf.circle(cx, cy, 24, fill=0, stroke=1)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.setFillColor(colors.HexColor("#A8E8DB"))
    pdf.drawString(43, height - 44, "FORMULATION SCIENCE  /  PREDICTIVE ANALYTICS")
    pdf.setFont("Helvetica-Bold", 22)
    pdf.setFillColor(paper)
    pdf.drawString(43, height - 80, "Phase Separation")
    pdf.drawString(43, height - 109, "Prediction Report")
    pdf.setFont("Helvetica", 9)
    pdf.setFillColor(colors.HexColor("#C1D4DE"))
    pdf.drawString(43, height - 134, "Machine-learning outcome assessment")

    issued = datetime.now(timezone.utc)
    report_id = issued.strftime("PS-%Y%m%d-%H%M%S-%f")[:-3]
    y = height - 190
    pdf.setFillColor(muted)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(43, y, "REPORT REFERENCE")
    pdf.drawString(325, y, "GENERATED (UTC)")
    pdf.setFillColor(ink)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(43, y - 17, report_id)
    pdf.drawString(325, y - 17, issued.strftime("%d %B %Y, %H:%M"))

    y -= 55
    pdf.setFillColor(ink)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(43, y, "Formulation input parameters")
    pdf.setFillColor(muted)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(43, y - 18, "Values submitted to the prediction model")
    y -= 38

    row_height = 45
    for i, field in enumerate(FIELDS):
        top = y - i * row_height
        if i % 2 == 0:
            pdf.setFillColor(pale)
            pdf.roundRect(42, top - 39, width - 84, 41, 6, fill=1, stroke=0)
        pdf.setFillColor(ink)
        pdf.setFont("Helvetica", 10.5)
        pdf.drawString(57, top - 24, field["label"])
        pdf.setFont("Helvetica-Bold", 11)
        value_text = f"{values[field['key']]:,.10g} {field['unit']}"
        pdf.drawRightString(width - 57, top - 24, value_text)

    y = y - row_height * len(FIELDS) - 18
    pdf.setFillColor(ink)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(43, y, "Model prediction")
    y -= 91

    outcome_color = warning if result["separated"] else teal
    outcome_bg = colors.HexColor("#FFF4EB") if result["separated"] else colors.HexColor("#EAF7F3")
    pdf.setFillColor(outcome_bg)
    pdf.roundRect(42, y - 1, width - 84, 65, 10, fill=1, stroke=0)
    pdf.setFillColor(outcome_color)
    pdf.setFont("Helvetica-Bold", 17)
    pdf.drawString(60, y + 35, result["label"])
    pdf.setFillColor(ink)
    pdf.setFont("Helvetica", 9.3)
    pdf.drawString(60, y + 14, result["summary"])

    footer_y = 77
    pdf.setStrokeColor(colors.HexColor("#DCE7EB"))
    pdf.line(43, footer_y + 28, width - 43, footer_y + 28)
    pdf.setFillColor(muted)
    pdf.setFont("Helvetica", 8)
    pdf.drawString(43, footer_y + 12, "This prediction is model-generated and is not a substitute for laboratory stability testing.")
    pdf.drawString(43, footer_y - 2, "Reliability depends on model validation and the similarity of input ranges to training data.")
    pdf.drawString(43, 35, "Phase Separation Prediction Application")
    pdf.drawRightString(width - 43, 35, "Page 1 of 1")
    pdf.save()
    buffer.seek(0)
    return buffer, issued


@app.post("/report")
def download_report():
    try:
        values = parse_form(request.form)
        # Always predict again: do not accept a user-editable hidden result.
        result = predict_outcome(values)
        pdf_bytes, issued = build_pdf(values, result)
        filename = f"phase_separation_prediction_{issued:%Y%m%d_%H%M%S}.pdf"
        # Printing opens the very same PDF inline; downloading saves it as a file.
        inline = request.form.get("mode") == "view"
        return send_file(pdf_bytes, mimetype="application/pdf", as_attachment=not inline, download_name=filename)
    except ValueError as exc:
        return show_form(values=request.form, error=str(exc), status=400)
    except RuntimeError as exc:
        return show_form(values=request.form, error=str(exc), status=503)
    except Exception:
        app.logger.exception("Report generation error")
        return show_form(values=request.form, error="Could not generate the PDF report.", status=500)


@app.get("/health")
def health():
    return jsonify(status="ready" if model is not None else "model_missing", model_loaded=model is not None), (200 if model is not None else 503)


if __name__ == "__main__":
    app.run(debug=False)
