import os
import io
import json
import base64
import sqlite3
import html
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from openai import OpenAI

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib import colors

try:
    from risk_engine import assess_risk, build_recommendation
except ImportError:
    from risk_engine import assess_risk

    def build_recommendation(result):
        level = str(result.get("risk_level", "Moderate"))
        if level.lower() in {"critical", "high"}:
            return "Restrict access where appropriate and arrange a qualified structural inspection before repair work."
        if level.lower() == "moderate":
            return "Arrange a professional inspection and monitor the affected area; select repair only after the cause is confirmed."
        return "Continue observation and arrange professional inspection if the defect grows, spreads, leaks, or becomes associated with deformation."


APP_TITLE = "STRUCTURE DOCTOR AI"
DB_PATH = "inspection_history.db"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# -----------------------------
# EXISTING DARK-FIELD UI
# -----------------------------
# Keep the original light page background and dark input controls.
# Functionality changes do not alter the visual theme.
st.markdown(
    """
    <style>
    .stApp { background: #f8fafc; }
    .main-title { font-size: 2.25rem; font-weight: 800; color: #334155; margin-bottom: 0.1rem; }
    .subtitle { color: #52759f; font-size: 1rem; margin-bottom: 1.3rem; }

    /* Keep form controls like the user's existing screen */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"],
    div[data-baseweb="textarea"] {
        background-color: #27272f !important;
        border-color: #27272f !important;
        color: #ffffff !important;
        border-radius: 9px !important;
    }
    div[data-baseweb="select"] *,
    div[data-baseweb="input"] *,
    div[data-baseweb="textarea"] * {
        color: #ffffff !important;
    }
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        background: #27272f !important;
        color: #ffffff !important;
    }
    div[data-baseweb="input"] input::placeholder,
    div[data-baseweb="textarea"] textarea::placeholder {
        color: #aeb3c2 !important;
        opacity: 1 !important;
    }
    div[data-baseweb="select"] svg { fill: #ffffff !important; }
    [data-testid="stNumberInput"] button {
        background: #27272f !important;
        color: #ffffff !important;
        border: none !important;
    }
    [data-testid="stNumberInput"] button:hover { background: #30303a !important; }
    [data-testid="stFileUploader"] section {
        background: #27272f !important;
        border-color: #27272f !important;
    }
    [data-testid="stFileUploader"] section * { color: #ffffff !important; }

    /* Dropdown menu */
    [data-baseweb="popover"] [role="option"] {
        background: #27272f !important;
        color: #ffffff !important;
    }
    [data-baseweb="popover"] [role="option"]:hover {
        background: #353540 !important;
    }

    .card {
        background: white; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 18px; margin-bottom: 14px; box-shadow: 0 2px 8px rgba(15,23,42,.04);
    }
    .section-title { color: #334155; font-size: 1.15rem; font-weight: 750; margin-bottom: 8px; }
    .muted { color: #64748b; font-size: .9rem; }
    .metric-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 16px; text-align: center; height: 100%;
    }
    .metric-label { color: #64748b; font-size: .82rem; }
    .metric-value { color: #0f172a; font-size: 1.45rem; font-weight: 800; margin-top: 3px; }
    .risk-box { border-radius: 14px; padding: 18px; background: #f8fafc; border: 1px solid #cbd5e1; }
    .repair-box { border-radius: 14px; padding: 16px; background: #f8fafc; border: 1px solid #cbd5e1; }
    .notice {
        background: #fff7ed; border: 1px solid #fed7aa; border-radius: 12px;
        padding: 14px; color: #9a3412;
    }
    .success-note {
        background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px;
        padding: 14px; color: #166534;
    }

    /* Keep all normal text visible on the light page background.
       These rules do NOT change the dark form-control theme above. */
    .stApp, .stApp p, .stApp span, .stApp label,
    .stApp .stMarkdown, .stApp .stCaption,
    .stApp [data-testid="stMarkdownContainer"] {
        color: #334155;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp h5, .stApp h6 {
        color: #334155 !important;
    }
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stWidgetLabel"] label,
    .stApp [data-testid="stCheckbox"] label,
    .stApp [data-testid="stRadio"] label,
    .stApp [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #334155 !important;
    }
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] * {
        color: #64748b !important;
    }
    .stApp .stAlert, .stApp .stAlert * {
        color: inherit;
    }
    div.stButton > button {
        border-radius: 9px; font-weight: 700; border: 1px solid #cbd5e1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def esc(value):
    return html.escape(str(value if value is not None else ""))


def render_html(content):
    st.markdown(content, unsafe_allow_html=True)


# -----------------------------
# DATABASE
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            structure_type TEXT,
            building_type TEXT,
            element TEXT,
            material TEXT,
            location TEXT,
            building_age TEXT,
            environment TEXT,
            crack_type TEXT,
            applied_load TEXT,
            load_unit TEXT,
            load_type TEXT,
            load_location TEXT,
            crack_detected INTEGER,
            damage_type TEXT,
            severity TEXT,
            risk_score INTEGER,
            risk_level TEXT,
            crack_length TEXT,
            crack_width TEXT,
            cause TEXT,
            solution TEXT,
            recommended_repair TEXT,
            urgency TEXT,
            ai_analysis TEXT
        )
        """
    )
    # Safe migration for older DBs.
    existing = {row[1] for row in cur.execute("PRAGMA table_info(inspections)").fetchall()}
    columns = {
        "building_type": "TEXT",
        "element": "TEXT",
        "material": "TEXT",
        "location": "TEXT",
        "building_age": "TEXT",
        "environment": "TEXT",
        "crack_type": "TEXT",
        "applied_load": "TEXT",
        "load_unit": "TEXT",
        "load_type": "TEXT",
        "load_location": "TEXT",
        "crack_detected": "INTEGER",
        "damage_type": "TEXT",
        "severity": "TEXT",
        "risk_score": "INTEGER",
        "risk_level": "TEXT",
        "crack_length": "TEXT",
        "crack_width": "TEXT",
        "cause": "TEXT",
        "solution": "TEXT",
        "recommended_repair": "TEXT",
        "urgency": "TEXT",
        "ai_analysis": "TEXT",
    }
    for name, dtype in columns.items():
        if name not in existing:
            cur.execute(f"ALTER TABLE inspections ADD COLUMN {name} {dtype}")
    conn.commit()
    conn.close()


init_db()


def save_inspection(inputs, ai_result, risk_result, cv_result):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO inspections (
            timestamp, structure_type, building_type, element, material, location,
            building_age, environment, crack_type, applied_load, load_unit, load_type,
            load_location, crack_detected, damage_type, severity, risk_score, risk_level,
            crack_length, crack_width, cause, solution, recommended_repair, urgency, ai_analysis
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            inputs.get("structure_type", ""),
            inputs.get("building_type", ""),
            inputs.get("element", ""),
            inputs.get("material", ""),
            inputs.get("location", ""),
            inputs.get("building_age", ""),
            inputs.get("environment", ""),
            inputs.get("crack_type", ""),
            inputs.get("applied_load", ""),
            inputs.get("load_unit", ""),
            inputs.get("load_type", ""),
            inputs.get("load_location", ""),
            int(bool(ai_result.get("crack_detected"))),
            ai_result.get("damage_type", ""),
            ai_result.get("severity", ""),
            int(risk_result.get("risk_score", ai_result.get("risk_score", 0))),
            risk_result.get("risk_level", ""),
            cv_result.get("length_text", ""),
            cv_result.get("width_text", ""),
            ai_result.get("cause", ""),
            ai_result.get("solution", ""),
            ai_result.get("recommended_repair", ""),
            ai_result.get("urgency", ""),
            json.dumps(ai_result, ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()


# -----------------------------
# OPENAI / AI VISION
# -----------------------------
def get_api_key():
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None
    if key:
        return str(key).strip()
    key = os.getenv("OPENAI_API_KEY")
    return str(key).strip() if key else None


def image_to_jpeg_bytes(image):
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def extract_json_from_response(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.replace("```json", "", 1).replace("```", "")
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def clean_ai_result(data):
    data = data if isinstance(data, dict) else {}
    severity = str(data.get("severity", "Moderate")).strip().title()
    if severity not in {"Low", "Moderate", "High", "Critical"}:
        severity = "Moderate"

    try:
        risk_score = int(float(data.get("risk_score", 0)))
    except Exception:
        risk_score = 0
    risk_score = max(0, min(100, risk_score))

    causes = data.get("possible_causes", [])
    if isinstance(causes, str):
        causes = [causes]
    if not isinstance(causes, list):
        causes = []
    causes = [str(x).strip() for x in causes if str(x).strip()]

    detected = bool(data.get("crack_detected", False))
    damage_type = str(data.get("damage_type", "No obvious crack/damage detected")).strip()

    cause = str(data.get("cause", "Undetermined from image")).strip()
    solution = str(data.get("solution", "Professional structural inspection before selecting a repair method.")).strip()
    repair = str(data.get("recommended_repair", "Professional inspection and repair-method selection based on confirmed cause.")).strip()
    urgency = str(data.get("urgency", "Professional inspection recommended.")).strip()

    if detected:
        if not cause:
            cause = "Possible cause cannot be confirmed from a photograph alone."
        if not solution:
            solution = "Confirm the cause through a qualified site inspection before repair."
        if not repair:
            repair = "Select the repair method only after the defect mechanism and structural condition are verified."
        if not urgency:
            urgency = "Arrange professional inspection; restrict access if damage appears severe or unstable."

    return {
        "available": bool(data.get("available", True)),
        "crack_detected": detected,
        "damage_type": damage_type,
        "location": str(data.get("location", "Not clearly identifiable from image")).strip(),
        "severity": severity,
        "risk_score": risk_score,
        "description": str(data.get("description", "No detailed visual description was returned.")).strip(),
        "visual_evidence": str(data.get("visual_evidence", "")).strip(),
        "possible_causes": causes,
        "cause": cause,
        "solution": solution,
        "recommended_repair": repair,
        "urgency": urgency,
        "recommendation": str(data.get("recommendation", "")).strip(),
    }


def ai_fallback(reason="AI analysis unavailable"):
    return clean_ai_result(
        {
            "available": False,
            "crack_detected": False,
            "damage_type": "AI analysis unavailable",
            "location": "Not available",
            "severity": "Moderate",
            "risk_score": 0,
            "description": reason,
            "visual_evidence": "Use the OpenCV screening result as a supplementary image-based indicator only.",
            "possible_causes": [],
            "cause": "Undetermined from image",
            "solution": "Professional structural inspection before selecting a repair method.",
            "recommended_repair": "Do not select a structural repair method from this automated result alone.",
            "urgency": "Professional inspection recommended if damage is suspected.",
            "recommendation": "Configure OPENAI_API_KEY in Streamlit Secrets for AI visual analysis.",
        }
    )


def analyze_image_with_ai(image, structure_context):
    api_key = get_api_key()
    if not api_key:
        return ai_fallback("OPENAI_API_KEY is not configured.")

    prompt = f"""
You are a structural-damage visual screening assistant for STRUCTURE DOCTOR AI.
This is a PRELIMINARY visual inspection only. Do not certify safety, structural capacity,
remaining strength, crack depth, reinforcement condition, or hidden damage from a photograph.
Do not claim that a structure is safe or unsafe solely from this image.

Inspect the supplied image and return ONLY valid JSON with these keys:
crack_detected, damage_type, location, severity, risk_score, description,
visual_evidence, possible_causes, cause, solution, recommended_repair, urgency, recommendation.

severity must be one of: Low, Moderate, High, Critical.
risk_score must be an integer from 0 to 100 and represent a preliminary visual concern score,
not a calculated structural capacity.

If a crack/damage is visible, EVERY detected defect must have:
1. cause
2. solution
3. recommended_repair
4. urgency

Important limitations:
- Do not infer crack depth from the photograph.
- Do not determine actual load capacity.
- Do not certify structural safety.
- Do not invent measurements that cannot be visually established.
- If the image is unclear, say so.
- If severe visible damage is present, recommend restricted access and prompt qualified structural inspection.
- Recommended repairs must be framed as preliminary guidance pending site verification.

Inspection context:
{json.dumps(structure_context, ensure_ascii=False)}
"""

    try:
        client = OpenAI(api_key=api_key)
        image_b64 = base64.b64encode(image_to_jpeg_bytes(image)).decode("utf-8")
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    ],
                }
            ],
        )
        parsed = extract_json_from_response(response.output_text)
        parsed["available"] = True
        return clean_ai_result(parsed)
    except Exception as exc:
        return ai_fallback(f"AI analysis failed: {exc}")


# -----------------------------
# OPENCV IMAGE SCREENING
# -----------------------------
def format_measurement(value, scale_mm_per_px=None):
    if scale_mm_per_px and scale_mm_per_px > 0:
        return f"{value * scale_mm_per_px:.2f} mm"
    return f"{value:.0f} px"


def classify_crack(length_px, width_px):
    if length_px <= 0:
        return "Not classified"
    ratio = length_px / max(width_px, 1)
    if ratio >= 20:
        return "Long / narrow crack-like feature"
    if ratio >= 8:
        return "Moderate linear crack-like feature"
    return "Short / wider damage-like feature"


def cv_analyze(image, scale_mm_per_px=None):
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
    _, dark_mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    edges = cv2.Canny(blur, 50, 150)
    combined = cv2.bitwise_or(dark_mask, edges)

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, clean_kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, clean_kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, 8)
    h, w = gray.shape
    image_area = h * w
    candidates = []

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area < max(20, image_area * 0.00003):
            continue
        if area > image_area * 0.15:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect < 2.0:
            continue
        score = area * min(aspect, 30)
        candidates.append((score, x, y, bw, bh, area))

    overlay = bgr.copy()
    mask = np.zeros_like(gray)

    if not candidates:
        return {
            "detected": False,
            "length_px": 0,
            "width_px": 0,
            "length_text": "Not detected",
            "width_text": "Not detected",
            "classification": "No strong linear feature detected",
            "overlay": Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)),
            "mask": Image.fromarray(mask),
        }

    candidates.sort(reverse=True)
    _, x, y, bw, bh, area = candidates[0]
    cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (40, 80, 220), 3)
    cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)

    length_px = float(max(bw, bh))
    width_proxy_px = float(max(1, min(bw, bh) * 0.18))

    return {
        "detected": True,
        "length_px": length_px,
        "width_px": width_proxy_px,
        "length_text": format_measurement(length_px, scale_mm_per_px),
        "width_text": format_measurement(width_proxy_px, scale_mm_per_px),
        "classification": classify_crack(length_px, width_proxy_px),
        "overlay": Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)),
        "mask": Image.fromarray(mask),
    }


# -----------------------------
# RISK ENGINE ADAPTER
# -----------------------------
def run_risk_engine(inputs, ai_result, cv_result):
    enriched = dict(inputs)
    enriched["ai_result"] = ai_result
    enriched["cv_result"] = {
        "detected": cv_result.get("detected", False),
        "length_px": cv_result.get("length_px", 0),
        "width_px": cv_result.get("width_px", 0),
    }

    try:
        result = assess_risk(enriched)
        if not isinstance(result, dict):
            result = {}
    except Exception as exc:
        # Safe local fallback so the app still runs if the project's risk_engine API differs.
        score = int(ai_result.get("risk_score", 0))
        if ai_result.get("severity") == "Critical":
            score = max(score, 85)
        elif ai_result.get("severity") == "High":
            score = max(score, 65)
        elif ai_result.get("severity") == "Moderate":
            score = max(score, 35)
        level = "Low" if score < 25 else "Moderate" if score < 50 else "High" if score < 75 else "Critical"
        result = {
            "risk_score": score,
            "risk_level": level,
            "severity": ai_result.get("severity", "Moderate"),
            "explanation": f"Fallback screening logic used because the risk engine could not be executed: {exc}",
            "recommendations": [],
            "safety_override": False,
        }

    score = result.get("risk_score", ai_result.get("risk_score", 0))
    try:
        score = int(float(score))
    except Exception:
        score = int(ai_result.get("risk_score", 0))
    score = max(0, min(100, score))

    level = str(result.get("risk_level", "")).strip() or (
        "Low" if score < 25 else "Moderate" if score < 50 else "High" if score < 75 else "Critical"
    )

    result["risk_score"] = score
    result["risk_level"] = level
    result["severity"] = result.get("severity") or ai_result.get("severity", "Moderate")
    result["explanation"] = result.get("explanation") or "Preliminary image/context screening only."
    result["recommendations"] = result.get("recommendations") or []
    return result


# -----------------------------
# PDF REPORT
# -----------------------------
def ptext(value, style):
    return Paragraph(esc(value).replace("\n", "<br/>"), style)


def make_pdf(inputs, image, cv_result, ai_result, risk_result):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=APP_TITLE,
        author=APP_TITLE,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], fontSize=20, leading=24,
        alignment=TA_CENTER, textColor=colors.HexColor("#0F172A"), spaceAfter=5,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontSize=9.5, leading=13,
        alignment=TA_CENTER, textColor=colors.HexColor("#64748B"), spaceAfter=14,
    )
    h_style = ParagraphStyle(
        "H", parent=styles["Heading2"], fontSize=12.5, leading=16,
        textColor=colors.HexColor("#0F172A"), spaceBefore=8, spaceAfter=7,
    )
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=9, leading=13,
        textColor=colors.HexColor("#334155"),
    )
    small = ParagraphStyle(
        "Small", parent=body, fontSize=8, leading=11,
        textColor=colors.HexColor("#64748B"),
    )

    story = [
        Paragraph(APP_TITLE, title_style),
        Paragraph("Preliminary Structural Damage Screening Report", subtitle_style),
        Paragraph(
            "IMPORTANT: This report is an image-based preliminary screening aid. It does not determine actual structural load capacity, certify safety, establish crack depth, or replace inspection by a qualified structural professional.",
            small,
        ),
        Spacer(1, 8),
    ]

    # Uploaded image, preserving aspect ratio.
    image_buffer = io.BytesIO(image_to_jpeg_bytes(image))
    max_w = 165 * mm
    max_h = 105 * mm
    iw, ih = image.size
    scale = min(max_w / max(iw, 1), max_h / max(ih, 1))
    story.append(RLImage(image_buffer, width=iw * scale, height=ih * scale))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Structure Information", h_style))
    info_rows = [
        ["Structure Type", inputs.get("structure_type", "")],
        ["Building Type", inputs.get("building_type", "")],
        ["Element", inputs.get("element", "")],
        ["Primary Material", inputs.get("material", "")],
        ["Location on Structure", inputs.get("location", "")],
        ["Building Age", inputs.get("building_age", "")],
        ["Environment", inputs.get("environment", "")],
        ["Observed Crack Type", inputs.get("crack_type", "")],
        ["Applied Load", f"{inputs.get('applied_load', '')} {inputs.get('load_unit', '')}".strip()],
        ["Load Type", inputs.get("load_type", "")],
        ["Load Location", inputs.get("load_location", "")],
    ]
    info_table = Table([[ptext(a, body), ptext(b, body)] for a, b in info_rows], colWidths=[55 * mm, 120 * mm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([info_table, Spacer(1, 9)])

    story.append(Paragraph("2. Image-Based Crack / Damage Screening", h_style))
    cv_rows = [
        ["OpenCV feature detected", "Yes" if cv_result.get("detected") else "No"],
        ["Image-based length", cv_result.get("length_text", "Not detected")],
        ["Image-based width proxy", cv_result.get("width_text", "Not detected")],
        ["Feature classification", cv_result.get("classification", "")],
    ]
    cv_table = Table([[ptext(a, body), ptext(b, body)] for a, b in cv_rows], colWidths=[55 * mm, 120 * mm])
    cv_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.extend([cv_table, Spacer(1, 9)])

    story.append(Paragraph("3. AI Visual Findings", h_style))
    ai_rows = [
        ["Crack / Damage Detected", "Yes" if ai_result.get("crack_detected") else "No"],
        ["Damage Type", ai_result.get("damage_type", "")],
        ["Location", ai_result.get("location", "")],
        ["Severity", ai_result.get("severity", "")],
        ["Preliminary Visual Risk Score", str(ai_result.get("risk_score", 0)) + "/100"],
        ["Description", ai_result.get("description", "")],
        ["Visual Evidence", ai_result.get("visual_evidence", "")],
    ]
    ai_table = Table([[ptext(a, body), ptext(b, body)] for a, b in ai_rows], colWidths=[55 * mm, 120 * mm])
    ai_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.extend([ai_table, Spacer(1, 7)])

    causes = ai_result.get("possible_causes", [])
    if causes:
        story.append(Paragraph("Possible Causes", h_style))
        for cause in causes:
            story.append(Paragraph("• " + esc(cause), body))
        story.append(Spacer(1, 5))

    story.append(Paragraph("4. Cause + Solution + Recommended Repair + Urgency", h_style))
    repair_rows = [
        ["Cause", ai_result.get("cause", "Undetermined from image")],
        ["Solution", ai_result.get("solution", "Professional structural inspection before selecting a repair method.")],
        ["Recommended Repair", ai_result.get("recommended_repair", "Professional inspection before repair selection.")],
        ["Urgency", ai_result.get("urgency", "Professional inspection recommended.")],
    ]
    repair_table = Table([[ptext(a, body), ptext(b, body)] for a, b in repair_rows], colWidths=[48 * mm, 127 * mm])
    repair_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([
        repair_table,
        Spacer(1, 7),
        Paragraph(
            "Repair guidance is preliminary and must not be treated as a certified structural repair design. Confirm the defect mechanism, dimensions, loading, reinforcement/detailing and site condition before repair.",
            small,
        ),
    ])

    story.append(Paragraph("5. Risk Assessment", h_style))
    risk_rows = [
        ["Risk Score", f"{risk_result.get('risk_score', 0)}/100"],
        ["Risk Level", risk_result.get("risk_level", "")],
        ["Severity", risk_result.get("severity", "")],
        ["Explanation", risk_result.get("explanation", "")],
    ]
    risk_table = Table([[ptext(a, body), ptext(b, body)] for a, b in risk_rows], colWidths=[55 * mm, 120 * mm])
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.extend([risk_table, Spacer(1, 8)])

    recommendations = risk_result.get("recommendations", [])
    if recommendations:
        story.append(Paragraph("6. Recommended Actions", h_style))
        for item in recommendations:
            story.append(Paragraph("• " + esc(item), body))
        story.append(Spacer(1, 5))

    if ai_result.get("recommendation"):
        story.append(Paragraph("AI Recommendation", h_style))
        story.append(Paragraph(esc(ai_result.get("recommendation")), body))

    story.append(Paragraph("7. IMPORTANT SAFETY NOTICE", h_style))
    story.append(
        Paragraph(
            "This application is a preliminary screening tool. A photograph and user-entered load information cannot establish actual structural capacity or certify safety. If there is significant cracking, spalling, exposed reinforcement, deformation, instability, falling material, or other potentially dangerous damage, keep people away from the affected area as appropriate and obtain an on-site assessment by a qualified structural professional.",
            body,
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("# 🏗️")
    st.markdown(f"## {APP_TITLE}")
    st.caption("Preliminary structural damage screening")
    page = st.radio(
        "Navigation",
        ["Home", "Analyze", "Track Change", "Inspection History"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("AI + OpenCV + Risk Engine + PDF + SQLite")


# -----------------------------
# HOME
# -----------------------------
if page == "Home":
    render_html(
        f"""
        <div class="main-title">{APP_TITLE}</div>
        <div class="subtitle">Professional preliminary screening for visible cracks and structural damage indicators.</div>
        <div class="card">
            <div class="section-title">What this version does</div>
            <div class="muted">Upload a structural image, provide inspection context and applied-load information, run image screening, receive AI visual findings, generate preliminary repair guidance, save the inspection, and export a professional PDF report.</div>
        </div>
        """
    )
    c1, c2, c3, c4 = st.columns(4)
    for col, title, text in [
        (c1, "📷 AI Vision", "Visual crack / damage screening"),
        (c2, "🔎 OpenCV", "Image-based feature detection"),
        (c3, "📊 Risk Engine", "Preliminary risk assessment"),
        (c4, "📄 PDF", "Cause + solution + repair + urgency"),
    ]:
        with col:
            render_html(f'<div class="metric-card"><div class="metric-value">{title}</div><div class="metric-label">{text}</div></div>')

    st.markdown("### Workflow")
    st.write("1. Enter structure information and applied load context.")
    st.write("2. Upload a clear photo of the affected area.")
    st.write("3. Run OpenCV + AI visual screening.")
    st.write("4. Review risk, cause, solution, recommended repair and urgency.")
    st.write("5. Download the PDF and optionally save the inspection to history.")
    st.markdown(
        '<div class="notice"><b>Safety:</b> The app is a preliminary screening aid. It does not certify structural safety or actual load capacity.</div>',
        unsafe_allow_html=True,
    )


# -----------------------------
# ANALYZE
# -----------------------------
elif page == "Analyze":
    render_html('<div class="main-title">Structural Damage Analysis</div><div class="subtitle">Enter context first, then upload a clear image for combined OpenCV and AI screening.</div>')

    with st.form("inspection_form"):
        st.markdown("### 1. Structure Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            structure_type = st.selectbox("Structure Type", ["Building", "Bridge", "Industrial Structure", "Retaining Wall", "Other"])
            building_type = st.selectbox("Building Type", ["Residential", "Commercial", "Industrial", "Institutional", "Infrastructure", "Other"])
            element = st.selectbox("Structural Element", ["Beam", "Column", "Slab", "Wall", "Foundation", "Floor", "Roof", "Other"])
        with c2:
            material = st.selectbox("Primary Material", ["Reinforced Concrete", "Concrete", "Steel", "Masonry", "Brick", "Stone", "Other"])
            location = st.text_input("Location on Structure", placeholder="e.g. underside of beam near support")
            building_age = st.text_input("Building / Structure Age", placeholder="e.g. 18 years")
        with c3:
            environment = st.selectbox("Environment", ["Indoor / Dry", "Outdoor", "Coastal", "Humid", "Industrial / Chemical", "Water-Exposed", "Unknown"])
            crack_type = st.selectbox("Observed Crack Type", ["Unknown / Need AI Assessment", "Vertical", "Horizontal", "Diagonal", "Map / Network", "Longitudinal", "Spalling / Concrete Damage", "Other"])
            scale = st.number_input("Optional scale (mm per pixel)", min_value=0.0, value=0.0, step=0.001, format="%.3f", help="Use only if you have a reliable image scale reference. Otherwise the app reports pixels.")

        st.markdown("### 2. Applied Load Context")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            applied_load = st.text_input("Applied Load", placeholder="e.g. 2500")
        with c2:
            load_unit = st.selectbox("Load Unit", ["kN", "N", "kg", "tonne", "kN/m", "kN/m²", "Other"])
        with c3:
            load_type = st.selectbox("Load Type", ["Dead Load", "Live Load", "Point Load", "Distributed Load", "Impact / Dynamic", "Unknown"])
        with c4:
            load_location = st.text_input("Load Location", placeholder="e.g. center of slab")

        st.markdown("### 3. Condition Indicators")
        c1, c2, c3 = st.columns(3)
        with c1:
            water_leakage = st.checkbox("Water leakage / dampness")
        with c2:
            rust_staining = st.checkbox("Rust staining / exposed reinforcement")
        with c3:
            deformation = st.checkbox("Visible deformation / displacement")

        uploaded = st.file_uploader("Upload structural image", type=["jpg", "jpeg", "png", "webp"])
        submitted = st.form_submit_button("🔍 Analyze Structure", use_container_width=True, type="primary")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded inspection image", use_container_width=True)

    if submitted:
        if not uploaded:
            st.error("Please upload a structural image before analysis.")
            st.stop()

        image = Image.open(uploaded).convert("RGB")
        scale_value = scale if scale > 0 else None
        inputs = {
            "structure_type": structure_type,
            "building_type": building_type,
            "element": element,
            "material": material,
            "location": location,
            "building_age": building_age,
            "environment": environment,
            "crack_type": crack_type,
            "applied_load": applied_load,
            "load_unit": load_unit,
            "load_type": load_type,
            "load_location": load_location,
            "water_leakage": water_leakage,
            "rust_staining": rust_staining,
            "deformation": deformation,
        }

        with st.spinner("Running OpenCV image screening..."):
            cv_result = cv_analyze(image, scale_value)

        with st.spinner("Running AI visual analysis..."):
            ai_result = analyze_image_with_ai(image, inputs)

        with st.spinner("Running risk assessment..."):
            risk_result = run_risk_engine(inputs, ai_result, cv_result)

        # Store in session for PDF / download / save.
        st.session_state["last_analysis"] = {
            "inputs": inputs,
            "image": image,
            "cv": cv_result,
            "ai": ai_result,
            "risk": risk_result,
        }

    if "last_analysis" in st.session_state:
        data = st.session_state["last_analysis"]
        inputs = data["inputs"]
        image = data["image"]
        cv_result = data["cv"]
        ai_result = data["ai"]
        risk_result = data["risk"]

        st.divider()
        render_html('<div class="section-title">Analysis Result</div>')

        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            (c1, "Risk Score", f"{risk_result.get('risk_score', 0)}/100"),
            (c2, "Risk Level", risk_result.get("risk_level", "Unknown")),
            (c3, "Severity", ai_result.get("severity", "Unknown")),
            (c4, "AI Status", "Available" if ai_result.get("available") else "Unavailable"),
        ]
        for col, label, value in metrics:
            with col:
                render_html(f'<div class="metric-card"><div class="metric-label">{esc(label)}</div><div class="metric-value">{esc(value)}</div></div>')

        if not ai_result.get("available"):
            st.warning("AI visual analysis is unavailable. OpenCV screening and the preliminary risk workflow are still shown. Add OPENAI_API_KEY to Streamlit Secrets to enable AI analysis.")

        if cv_result.get("detected"):
            st.markdown("### OpenCV Image Screening")
            a, b = st.columns(2)
            with a:
                st.image(cv_result["overlay"], caption="Detected image-based feature", use_container_width=True)
            with b:
                st.image(cv_result["mask"], caption="Detection mask", use_container_width=True)
            st.write(f"**Image-based length:** {cv_result.get('length_text')}")
            st.write(f"**Image-based width proxy:** {cv_result.get('width_text')}")
            st.caption("These are image-processing measurements/proxies. They are not certified crack dimensions unless a reliable scale and field verification are available.")
        else:
            st.info("OpenCV did not detect a strong linear crack-like feature. This does not prove that no damage exists.")

        st.markdown("### AI Visual Findings")
        render_html(
            f"""
            <div class="card">
                <p><b>Damage type:</b> {esc(ai_result.get('damage_type'))}</p>
                <p><b>Location:</b> {esc(ai_result.get('location'))}</p>
                <p><b>Description:</b> {esc(ai_result.get('description'))}</p>
                <p><b>Visual evidence:</b> {esc(ai_result.get('visual_evidence'))}</p>
            </div>
            """
        )

        if ai_result.get("possible_causes"):
            st.markdown("### Possible Causes")
            for cause in ai_result["possible_causes"]:
                st.write(f"• {cause}")

        st.markdown("### Cause + Solution + Recommended Repair + Urgency")
        repair_data = [
            ("Cause", ai_result.get("cause", "Undetermined from image")),
            ("Solution", ai_result.get("solution", "Professional structural inspection before selecting a repair method.")),
            ("Recommended Repair", ai_result.get("recommended_repair", "Professional inspection before repair selection.")),
            ("Urgency", ai_result.get("urgency", "Professional inspection recommended.")),
        ]
        for title, value in repair_data:
            render_html(
                f'<div class="repair-box"><b>{esc(title)}</b><br/><span class="muted">{esc(value)}</span></div>'
            )

        st.markdown("### Risk Assessment")
        render_html(
            f"""
            <div class="risk-box">
                <b>Risk level:</b> {esc(risk_result.get('risk_level'))}<br/>
                <b>Risk score:</b> {esc(risk_result.get('risk_score'))}/100<br/><br/>
                <b>Explanation:</b> {esc(risk_result.get('explanation'))}
            </div>
            """
        )

        if risk_result.get("recommendations"):
            st.markdown("### Recommended Actions")
            for recommendation in risk_result["recommendations"]:
                st.write(f"• {recommendation}")

        if ai_result.get("recommendation"):
            st.info(ai_result["recommendation"])

        st.markdown("### Export / Save")
        pdf_bytes = make_pdf(inputs, image, cv_result, ai_result, risk_result)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"structure_doctor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with c2:
            if st.button("💾 Save Inspection to History", use_container_width=True):
                save_inspection(inputs, ai_result, risk_result, cv_result)
                st.success("Inspection saved to SQLite history.")

        st.markdown(
            '<div class="notice"><b>Safety notice:</b> Applied-load input is contextual information only. Entering a load value does not establish that the structure can safely carry that load.</div>',
            unsafe_allow_html=True,
        )


# -----------------------------
# TRACK CHANGE
# -----------------------------
elif page == "Track Change":
    render_html('<div class="main-title">Track Change</div><div class="subtitle">Compare two inspection images using the same image-processing workflow.</div>')
    before_file = st.file_uploader("Before image", type=["jpg", "jpeg", "png", "webp"], key="before")
    after_file = st.file_uploader("After image", type=["jpg", "jpeg", "png", "webp"], key="after")

    if before_file and after_file:
        before = Image.open(before_file).convert("RGB")
        after = Image.open(after_file).convert("RGB")
        c1, c2 = st.columns(2)
        with c1:
            st.image(before, caption="Before", use_container_width=True)
        with c2:
            st.image(after, caption="After", use_container_width=True)

        before_cv = cv_analyze(before)
        after_cv = cv_analyze(after)
        before_len = before_cv.get("length_px", 0)
        after_len = after_cv.get("length_px", 0)
        change = after_len - before_len

        m1, m2, m3 = st.columns(3)
        with m1:
            render_html(f'<div class="metric-card"><div class="metric-label">Before feature length</div><div class="metric-value">{esc(before_cv.get("length_text"))}</div></div>')
        with m2:
            render_html(f'<div class="metric-card"><div class="metric-label">After feature length</div><div class="metric-value">{esc(after_cv.get("length_text"))}</div></div>')
        with m3:
            render_html(f'<div class="metric-card"><div class="metric-label">Pixel change proxy</div><div class="metric-value">{change:+.0f} px</div></div>')

        st.warning("Track Change is an image-based comparison. Different camera angle, distance, lighting, cropping, or perspective can change the measured proxy even when the physical condition has not changed.")


# -----------------------------
# HISTORY
# -----------------------------
elif page == "Inspection History":
    render_html('<div class="main-title">Inspection History</div><div class="subtitle">Saved inspections from the local SQLite database.</div>')
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """
        SELECT id, timestamp, structure_type, element, material, risk_score,
               risk_level, severity, crack_length, crack_width, cause, urgency
        FROM inspections ORDER BY id DESC
        """
    ).fetchall()
    conn.close()

    if not rows:
        st.info("No saved inspections yet. Run an analysis and choose Save Inspection to History.")
    else:
        import pandas as pd
        df = pd.DataFrame(rows, columns=[
            "ID", "Timestamp", "Structure", "Element", "Material", "Risk Score",
            "Risk Level", "Severity", "Length", "Width", "Cause", "Urgency"
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
