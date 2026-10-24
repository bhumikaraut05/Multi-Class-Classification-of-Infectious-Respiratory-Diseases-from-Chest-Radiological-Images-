"""Locally generated, template-based report text and PDF.  No external services; nothing is invented:
every number comes from the real prediction response."""
from __future__ import annotations

import base64
import io
from datetime import datetime

DISCLAIMER = ("This is an academic research prototype. The output is an AI classification result with a "
              "model-estimated confidence. It is not a diagnosis, is not a medical device, and is not a substitute "
              "for evaluation by a qualified radiologist or physician.")

MEANING = {
    "COVID-19": "the training images labelled COVID-19",
    "Lung-Opacity": "the training images labelled Lung Opacity (non-COVID lung opacity)",
    "Normal": "the training images labelled Normal",
    "Viral Pneumonia": "the training images labelled Viral Pneumonia",
    "Tuberculosis": "the training images labelled Tuberculosis",
}

NEXT_STEPS = [
    "Review this result with a qualified radiologist or physician; the image should be interpreted by a professional.",
    "Obtain professional medical evaluation if you have symptoms or concerns, whatever this result says.",
    "Seek urgent medical attention for severe symptoms such as difficulty breathing, chest pain or confusion.",
    "Do not delay emergency care because of an AI prediction.",
]

LIMITATIONS = [
    "The model only chooses among five labels (COVID-19, Lung Opacity, Normal, Viral Pneumonia, Tuberculosis). "
    "Other conditions, or combinations of conditions, are not represented.",
    "It does not locate lesions, grade severity, or recommend treatment, medication or surgery.",
    "Performance on images from other hospitals, scanners or patient groups than the training data is unknown and may be lower.",
    "Confidence is the model's own estimate and can be high even when the prediction is wrong.",
]


def interpretation(pred: str, conf: float, low: bool) -> str:
    s = (f"The model's output for this image is closest to {MEANING.get(pred, pred)}, with an estimated confidence of "
         f"{conf:.1%}. This describes similarity to patterns the model learned; it is not a confirmed finding.")
    if pred == "Normal":
        s += " A Normal result does not rule out disease."
    if low:
        s += " The confidence is relatively low, so the result is uncertain."
    return s


def build_sections(r: dict, expl_available: bool, expl_note: str) -> list[dict]:
    probs = sorted(r["probabilities"].items(), key=lambda kv: -kv[1])
    return [
        {"title": "Analysis summary", "items": [
            f"Report ID: {r['prediction_id']}", f"Generated: {r['created_at']}", f"Image file: {r['filename']}",
            "Analysis status: completed", f"Model: {r['model']['name']} (version {r['model']['version']})",
            f"AI classification result: {r['prediction']}", f"Model-estimated confidence: {r['confidence']:.1%}"]},
        {"title": "Class probabilities", "items": [f"{k}: {v:.1%}" for k, v in probs]},
        {"title": "Image preprocessing", "items": [s["name"] for s in r["preprocessing_steps"]]},
        {"title": "What this result means", "body": interpretation(r["prediction"], r["confidence"], r["low_confidence"])},
        {"title": "Explainability", "body": expl_note if expl_available else "Not available for this prediction. " + expl_note},
        {"title": "Image-quality notes", "items": [w["message"] for w in r["quality_warnings"]] or ["No quality issues detected by the automatic checks (these checks are limited)."]},
        {"title": "Limitations", "items": LIMITATIONS},
        {"title": "Suggested next steps", "items": NEXT_STEPS},
        {"title": "Disclaimer", "body": DISCLAIMER},
    ]


def _img(data_url: str | None, width: float):
    from reportlab.platypus import Image
    if not data_url:
        return None
    raw = base64.b64decode(data_url.split(",", 1)[1])
    im = Image(io.BytesIO(raw))
    ratio = im.imageHeight / im.imageWidth
    im.drawWidth, im.drawHeight = width, width * ratio
    return im


def make_pdf(r: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=colors.HexColor("#1d4ed8"), fontSize=20)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=colors.HexColor("#1e3a8a"), spaceBefore=10)
    body = ss["BodyText"]
    esc = lambda t: str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")  # noqa: E731
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title=f"Chest X-ray AI report {r['prediction_id']}")
    story = [Paragraph("AI-Assisted Chest X-ray Analysis Report", h1),
             Paragraph("Academic research prototype - not a diagnosis", ss["Italic"]), Spacer(1, 6)]
    expl = r["explanation"]
    for sec in r["report"]:
        story.append(Paragraph(esc(sec["title"]), h2))
        if sec["title"] == "Class probabilities":
            rows = [["Class", "Probability", ""]]
            for k, v in sorted(r["probabilities"].items(), key=lambda kv: -kv[1]):
                rows.append([k, f"{v:.1%}", "#" * max(1, round(v * 30)) if v > 0.005 else ""])
            t = Table(rows, colWidths=[50 * mm, 28 * mm, 80 * mm])
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                                   ("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#94a3b8")),
                                   ("TEXTCOLOR", (2, 1), (2, -1), colors.HexColor("#2563eb"))]))
            story.append(t)
        elif sec["title"] == "Explainability":
            imgs = [_img(r["processed_image"], 52 * mm), _img(expl.get("overlay"), 52 * mm)]
            imgs = [i for i in imgs if i]
            if imgs:
                cap = ["Model input (preprocessed)"] + (["Explanation overlay"] if len(imgs) > 1 else [])
                t = Table([imgs, cap], colWidths=[58 * mm] * len(imgs))
                t.setStyle(TableStyle([("FONTSIZE", (0, 1), (-1, 1), 8), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
                story.append(t)
            story.append(Paragraph(esc(sec["body"] or ""), body))
        elif sec.get("items"):
            for it in sec["items"]:
                story.append(Paragraph("&bull; " + esc(it), body))
        else:
            story.append(Paragraph(esc(sec.get("body") or ""), body))
    doc.build(story)
    return buf.getvalue()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
