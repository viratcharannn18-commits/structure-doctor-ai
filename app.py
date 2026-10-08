import os
import io
import json
import base64
import sqlite3
import textwrap
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from openai import OpenAI

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
)

from risk_engine import assess_risk, build_recommendation


APP_TITLE = "STRUCTURE DOCTOR AI"
DB_PATH = "inspection_history.db"


# ============================================================
# PAGE / THEME
# ============================================================
st.set_page_config(
    page_title="Structure Doctor AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #F4F7FA;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    .hero {
        background: linear-gradient(135deg, #123B4A 0%, #1F6175 55%, #2F6F8F 100%);
        border-radius: 22px;
        padding: 42px 46px;
        margin-bottom: 24px;
        box-shadow: 0 14px 35px rgba(18, 59, 74, .16);
        color: white;
    }

    .hero-kicker {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        opacity: .84;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: 42px;
        line-height: 1.05;
        font-weight: 800;
        margin: 0;
    }

    .hero-subtitle {
        font-size: 17px;
        line-height: 1.55;
        margin-top: 15px;
        max-width: 900px;
        opacity: .94;
    }

    .hero-badge {
        display: inline-block;
        margin-top: 22px;
        padding: 8px 13px;
        border: 1px solid rgba(255,255,255,.28);
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
    }

    .section {
        background: white;
        border: 1px solid #D9E2E8;
        border-radius: 18px;
        padding: 26px 28px;
        margin: 18px 0;
        box-shadow: 0 7px 22px rgba(31,41,51,.055);
    }

    .section-title {
        color: #123B4A;
        font-size: 23px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .section-subtitle {
        color: #64748B;
        font-size: 13px;
        margin-bottom: 18px;
    }

    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    .stSelectbox label,
    .stNumberInput label,
    .stTextInput label,
    .stTextArea label,
    .stFileUploader label,
    .stSlider label,
    .stRadio label {
        color: #123B4A !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }

    [data-testid="stWidgetLabel"] small,
    [data-testid="InputInstructions"],
    [data-testid="stCaptionContainer"] {
        color: #64748B !important;
        opacity: 1 !important;
    }

    .stSelectbox > div > div,
    .stNumberInput > div > div,
    .stTextInput > div > div,
    .stTextArea > div > div {
        border-radius: 10px;
    }

    .step-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
    }

    .step-card {
        background: #F7F9FB;
        border: 1px solid #D9E2E8;
        border-radius: 14px;
        padding: 18px;
        min-height: 130px;
    }

    .step-number {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: #2F6F8F;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .step-title {
        color: #1F2933;
        font-weight: 700;
        font-size: 14px;
    }

    .step-text {
        color: #64748B;
        font-size: 12px;
        line-height: 1.5;
        margin-top: 5px;
    }

    .kpi-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
    }

    .kpi {
        background: white;
        border: 1px solid #D9E2E8;
        border-radius: 15px;
        padding: 19px;
    }

    .kpi .label {
        color: #64748B;
        font-size: 12px;
        font-weight: 600;
    }

    .kpi .value {
        color: #1F2933 !important;
        font-size: 25px;
        font-weight: 800;
        margin-top: 5px;
    }

    .result-card {
        background: #FFFFFF;
        border: 1px solid #D9E2E8;
        border-radius: 17px;
        padding: 23px;
        height: 100%;
    }

    .result-title {
        color: #123B4A;
        font-weight: 800;
        font-size: 17px;
        margin-bottom: 10px;
    }

    .callout, .warning, .danger, .good {
        border-radius: 14px;
        padding: 17px;
        margin: 12px 0;
    }

    .callout {
        background: #EAF4F8;
        border: 1px solid #B9D8E4;
    }

    .warning {
        background: #FFF7E6;
        border: 1px solid #F1D69B;
    }

    .danger {
        background: #FDECEC;
        border: 1px solid #E7B6B6;
    }

    .good {
        background: #EDF8F0;
        border: 1px solid #B9D9C0;
    }

    .callout *, .warning *, .danger *, .good * {
        color: #1F2933 !important;
    }

    .hero, .hero * {
        color: #FFFFFF !important;
    }

    section[data-testid="stSidebar"] {
        background: #123B4A;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .stButton > button {
        color: #1F2933 !important;
        border-radius: 10px;
        border: 1px solid #2F6F8F;
        font-weight: 700;
    }

    .stButton > button p {
        color: #1F2933 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #64748B !important;
    }

    [data-testid="stMetricValue"] {
        color: #1F2933 !important;
    }

    .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: #EAF4F8;
        color: #1F6175 !important;
        font-size: 11px;
        font-weight: 700;
        margin-right: 6px;
    }

    .footer {
        text-align: center;
        color: #718096;
        font-size: 11px;
        padding: 28px 10px 10px;
    }

    @media (max-width: 850px) {
        .step-row, .kpi-row {
            grid-template-columns: repeat(2, 1fr);
        }
        .hero-title {
            font-size: 32px;
        }
    }

    @media (max-width: 560px) {
        .step-row, .kpi-row {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_html(html: str):
    """Render HTML without leading indentation becoming a Markdown code block."""
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)


# ============================================================
# DATABASE
# ============================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            structure_type TEXT,
            building_type TEXT,
            element TEXT,
            location_on_structure TEXT,
            age TEXT,
            material TEXT,
            environment TEXT,
            applied_load REAL,
            load_unit TEXT,
            load_type TEXT,
            load_location TEXT,
            crack_type TEXT,
            severity TEXT,
            score INTEGER,
            confidence REAL,
            length_value TEXT,
            width_value TEXT,
            ai_available INTEGER,
            recommendation TEXT,
            cause TEXT,
            solution TEXT,
            recommended_repair TEXT,
            urgency TEXT
        )
        """
    )

    existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(inspections)").fetchall()
    }

    additions = {
        "structure_type": "TEXT",
        "building_type": "TEXT",
        "location_on_structure": "TEXT",
        "environment": "TEXT",
        "applied_load": "REAL",
        "load_unit": "TEXT",
        "load_type": "TEXT",
        "load_location": "TEXT",
        "ai_available": "INTEGER DEFAULT 0",
        "cause": "TEXT",
        "solution": "TEXT",
        "recommended_repair": "TEXT",
        "urgency": "TEXT",
    }

    for column, dtype in additions.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE inspections ADD COLUMN {column} {dtype}")

    conn.commit()
    conn.close()


def save_inspection(info, result, confidence, length_text, width_text, recommendation, ai_result):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO inspections (
            timestamp,
            structure_type,
            building_type,
            element,
            location_on_structure,
            age,
            material,
            environment,
            applied_load,
            load_unit,
            load_type,
            load_location,
            crack_type,
            severity,
            score,
            confidence,
            length_value,
            width_value,
            ai_available,
            recommendation,
            cause,
            solution,
            recommended_repair,
            urgency
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            info.get("structure_type", ""),
            info.get("building_type", ""),
            info.get("element", ""),
            info.get("location_on_structure", ""),
            info.get("age", ""),
            info.get("material", ""),
            info.get("environment", ""),
            float(info.get("applied_load", 0) or 0),
            info.get("load_unit", ""),
            info.get("load_type", ""),
            info.get("load_location", ""),
            info.get("crack_type", ""),
            result.get("severity", ""),
            int(result.get("score", result.get("risk_score", 0)) or 0),
            float(confidence or 0),
            length_text,
            width_text,
            1 if ai_result.get("available") else 0,
            recommendation.get("next_step", ""),
            ai_result.get("cause", ""),
            ai_result.get("solution", ""),
            ai_result.get("recommended_repair", ""),
            ai_result.get("urgency", ""),
        ),
    )
    conn.commit()
    conn.close()


init_db()


# ============================================================
# OPENAI VISION
# ============================================================
def get_api_key():
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        key = ""

    if not key:
        key = os.getenv("OPENAI_API_KEY", "")

    return key.strip()


def image_to_jpeg_bytes(image: Image.Image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()


def clean_ai_result(data):
    if not isinstance(data, dict):
        data = {}

    possible_causes = data.get("possible_causes", [])
    if isinstance(possible_causes, str):
        possible_causes = [possible_causes]
    if not isinstance(possible_causes, list):
        possible_causes = []

    detected = bool(data.get("crack_detected", data.get("detected", False)))

    return {
        "available": bool(data.get("available", True)),
        "crack_detected": detected,
        "damage_type": str(data.get("damage_type", "Crack-like damage")),
        "location": str(data.get("location", "Not determined from image")),
        "severity": str(data.get("severity", "Low")),
        "risk_score": int(float(data.get("risk_score", 0) or 0)),
        "description": str(
            data.get(
                "description",
                "No additional visual description was returned.",
            )
        ),
        "visual_evidence": str(
            data.get(
                "visual_evidence",
                data.get(
                    "description",
                    "No additional visual evidence was returned.",
                ),
            )
        ),
        "possible_causes": [str(x) for x in possible_causes if str(x).strip()],
        "cause": str(data.get("cause", "Undetermined from image")),
        "solution": str(
            data.get(
                "solution",
                "Professional structural inspection before selecting a repair method.",
            )
        ),
        "recommended_repair": str(
            data.get(
                "recommended_repair",
                "Have a qualified structural professional inspect the affected area before repair selection.",
            )
        ),
        "urgency": str(
            data.get(
                "urgency",
                "Professional inspection recommended.",
            )
        ),
        "recommendation": str(
            data.get(
                "recommendation",
                "Use the result as preliminary screening only.",
            )
        ),
    }


def analyze_image_with_ai(image: Image.Image, structure_context: dict):
    api_key = get_api_key()

    fallback = clean_ai_result(
        {
            "available": False,
            "crack_detected": False,
            "damage_type": "AI unavailable",
            "location": "Not determined",
            "severity": "Low",
            "risk_score": 0,
            "description": "AI visual analysis was not available.",
            "visual_evidence": "OpenCV screening is available; AI visual interpretation was not returned.",
            "possible_causes": [],
            "cause": "Undetermined from image",
            "solution": "Professional structural inspection before selecting a repair method.",
            "recommended_repair": "Use qualified structural inspection to determine the appropriate repair.",
            "urgency": "Professional inspection recommended.",
            "recommendation": "Review the OpenCV screening and obtain professional inspection for significant or uncertain damage.",
        }
    )

    if not api_key:
        return fallback

    try:
        client = OpenAI(api_key=api_key)
        image_b64 = base64.b64encode(image_to_jpeg_bytes(image)).decode("utf-8")

        prompt = f"""
You are an AI-assisted structural visual screening assistant.

This is a PRELIMINARY visual inspection only. Do not certify structural safety,
do not calculate actual load capacity, do not infer hidden reinforcement,
and do not claim crack depth from a normal photograph.

Structural context:
{json.dumps(structure_context, indent=2)}

Inspect the uploaded image and return ONLY valid JSON with these keys:
{{
  "crack_detected": true or false,
  "damage_type": "...",
  "location": "...",
  "severity": "Low" | "Moderate" | "High" | "Critical",
  "risk_score": 0-100,
  "description": "...",
  "visual_evidence": "...",
  "possible_causes": ["...", "..."],
  "cause": "...",
  "solution": "...",
  "recommended_repair": "...",
  "urgency": "...",
  "recommendation": "..."
}}

Rules:
- Base observations only on visible evidence and supplied context.
- Do not state that the structure is safe.
- Do not state that the structure will fail.
- Do not invent measurements.
- If no reliable crack-like damage is visible, say so.
- If visible damage could be significant, recommend restricted access where appropriate
  and professional structural inspection.
- For every detected crack/damage region, provide Cause, Solution,
  Recommended Repair, and Urgency.
- Applied load is contextual information only. Never treat a user-entered load
  value as proof that the structure is safe or adequate.
"""

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

        parsed = json.loads(response.output_text)
        parsed["available"] = True
        return clean_ai_result(parsed)

    except Exception as exc:
        fallback["error"] = str(exc)
        return fallback


# ============================================================
# OPENCV
# ============================================================
def cv_analyze(image: Image.Image):
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, mask = cv2.threshold(
        blackhat,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    edges = cv2.Canny(gray, 50, 150)
    combined = cv2.bitwise_or(mask, edges)

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        combined, 8
    )

    h, w = gray.shape
    image_area = max(h * w, 1)
    candidates = []

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]

        if area < max(20, image_area * 0.00005):
            continue

        if area > image_area * 0.35:
            continue

        aspect = max(bw, bh) / max(min(bw, bh), 1)

        if aspect < 2.0:
            continue

        candidates.append((area, x, y, bw, bh, aspect))

    if not candidates:
        return {
            "detected": False,
            "confidence": 0.0,
            "length_px": 0.0,
            "width_px": 0.0,
            "area_px": 0.0,
            "overlay": image.convert("RGB"),
            "mask": Image.fromarray(np.zeros_like(gray)),
        }

    candidates.sort(
        key=lambda item: item[5] * np.sqrt(max(item[0], 1)),
        reverse=True,
    )

    area, x, y, bw, bh, aspect = candidates[0]

    length_px = float(max(bw, bh))
    width_px = float(max(1.0, min(bw, bh) * 0.18))
    width_px = min(width_px, max(length_px * 0.25, 1.0))

    confidence = min(
        98.0,
        max(
            55.0,
            55.0
            + min(aspect / 12.0, 1.0) * 25.0
            + min(area / (image_area * 0.03), 1.0) * 18.0,
        ),
    )

    overlay = rgb.copy()

    cv2.rectangle(
        overlay,
        (x, y),
        (x + bw, y + bh),
        (47, 111, 143),
        4,
    )

    cv2.putText(
        overlay,
        "Detected region",
        (x, max(25, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (47, 111, 143),
        2,
        cv2.LINE_AA,
    )

    roi_mask = np.zeros_like(gray)
    roi_mask[y:y + bh, x:x + bw] = combined[y:y + bh, x:x + bw]

    return {
        "detected": True,
        "confidence": float(confidence),
        "length_px": length_px,
        "width_px": width_px,
        "area_px": float(area),
        "overlay": Image.fromarray(overlay),
        "mask": Image.fromarray(roi_mask),
    }


def format_measurement(value, scale):
    if scale and scale > 0:
        return f"{value * scale:.2f} mm"
    return f"{value:.0f} px"


def classify_crack(cv_result):
    length = cv_result.get("length_px", 0)
    width = cv_result.get("width_px", 0)

    if length <= 0:
        return "Auto / unknown"

    if length > width * 7:
        return "Vertical / Horizontal"

    return "Random / irregular"


# ============================================================
# PDF
# ============================================================
def make_pdf(
    info,
    result,
    recommendation,
    confidence,
    length_text,
    width_text,
    ai_result,
    image,
):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "Title2",
        parent=styles["Title"],
        fontSize=22,
        leading=26,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#123B4A"),
    )

    section = ParagraphStyle(
        "Section2",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#123B4A"),
        spaceBefore=5 * mm,
        spaceAfter=2 * mm,
    )

    body = ParagraphStyle(
        "Body2",
        parent=styles["BodyText"],
        fontSize=9.2,
        leading=13,
        textColor=colors.HexColor("#1F2933"),
    )

    small = ParagraphStyle(
        "Small2",
        parent=body,
        fontSize=8,
        leading=11,
    )

    story = [
        Paragraph("STRUCTURE DOCTOR AI", title),
        Spacer(1, 4 * mm),
        Paragraph(
            "AI-Assisted Crack Detection & Preliminary Structural Assessment",
            body,
        ),
        Spacer(1, 3 * mm),
        Paragraph(
            "<b>Preliminary screening only — not a structural safety certificate.</b>",
            body,
        ),
        Spacer(1, 6 * mm),
    ]

    try:
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="JPEG", quality=85)
        image_buffer.seek(0)
        display_height = min(
            105 * mm,
            165 * mm * image.height / max(image.width, 1),
        )
        story.append(
            RLImage(
                image_buffer,
                width=165 * mm,
                height=display_height,
            )
        )
        story.append(Spacer(1, 5 * mm))
    except Exception:
        pass

    story.append(Paragraph("1. Inspection Information", section))

    data = [
        ["Inspection", datetime.now().strftime("%d %b %Y, %H:%M")],
        ["Structure type", str(info.get("structure_type", ""))],
        ["Building type", str(info.get("building_type", ""))],
        ["Structural element", str(info.get("element", ""))],
        ["Location on structure", str(info.get("location_on_structure", ""))],
        ["Building age", str(info.get("age", ""))],
        ["Primary material", str(info.get("material", ""))],
        ["Environment", str(info.get("environment", ""))],
        [
            "Applied load",
            f'{info.get("applied_load", 0)} {info.get("load_unit", "")} '
            f'({info.get("load_type", "")}, {info.get("load_location", "")})',
        ],
        ["Observed crack pattern", str(info.get("crack_type", ""))],
        ["Estimated length", length_text],
        ["Estimated width", width_text],
    ]

    table = Table(data, colWidths=[55 * mm, 115 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF4F8")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1F2933")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9E2E8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(table)

    story.append(Paragraph("2. Visual / AI Findings", section))

    findings = [
        ["AI availability", "Available" if ai_result.get("available") else "Unavailable"],
        ["Damage detected", "Yes" if ai_result.get("crack_detected") else "No / uncertain"],
        ["Damage type", ai_result.get("damage_type", "")],
        ["Visible location", ai_result.get("location", "")],
        ["AI severity", ai_result.get("severity", "")],
        ["AI risk score", str(ai_result.get("risk_score", 0))],
        ["Visual description", ai_result.get("description", "")],
    ]

    ft = Table(findings, colWidths=[45 * mm, 125 * mm])
    ft.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F7F9FB")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9E2E8")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(ft)

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("<b>Possible causes</b>", body))
    for cause in ai_result.get("possible_causes", []):
        story.append(Paragraph(f"• {cause}", body))

    story.append(Paragraph("3. Crack Cause & Recommended Repair", section))

    repair_data = [
        ["Cause", ai_result.get("cause", "Undetermined from image")],
        ["Solution", ai_result.get("solution", "")],
        ["Recommended Repair", ai_result.get("recommended_repair", "")],
        ["Urgency", ai_result.get("urgency", "")],
    ]

    rt = Table(repair_data, colWidths=[48 * mm, 122 * mm])
    rt.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF4F8")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(rt)

    story.append(Paragraph("4. Risk Assessment", section))
    score = result.get("score", result.get("risk_score", 0))
    risk_data = [
        ["Preliminary severity", str(result.get("severity", ""))],
        ["Screening score", str(score)],
        ["OpenCV confidence", f"{confidence:.0f}%"],
        ["Visual evidence", str(result.get("visual_evidence", ""))],
    ]
    risk_table = Table(risk_data, colWidths=[55 * mm, 115 * mm])
    risk_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F7F9FB")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9E2E8")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(risk_table)

    story.append(Paragraph("5. Recommended Actions", section))
    story.append(Paragraph(recommendation.get("next_step", ""), body))
    for item in recommendation.get("safe_actions", []):
        story.append(Paragraph(f"• {item}", body))

    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            "<b>Safety notice:</b> Applied load entered by the user is contextual "
            "information only. It is not treated as proof of structural adequacy. "
            "This system does not determine actual structural strength, remaining "
            "load capacity, reinforcement condition, crack depth, or final cause. "
            "Significant, progressing, displaced, or uncertain damage should be "
            "assessed by a qualified structural professional.",
            small,
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🏗️ Structure Doctor")
    st.caption("AI-assisted preliminary screening")

    page = st.radio(
        "Navigate",
        ["Home", "Analyze", "Track Change", "Inspection History"],
    )

    st.markdown("---")
    st.markdown("### System")
    st.markdown(
        '<span class="pill">OpenCV</span>'
        '<span class="pill">OpenAI Vision</span>'
        '<span class="pill">SQLite</span>',
        unsafe_allow_html=True,
    )


# ============================================================
# HOME
# ============================================================
if page == "Home":
    render_html(
        """
        <div class="hero">
            <div class="hero-kicker">AI-Assisted Structural Screening</div>
            <div class="hero-title">STRUCTURE<br>DOCTOR AI</div>
            <div class="hero-subtitle">
                A visual screening prototype that combines structural context,
                OpenCV image features and optional AI vision analysis to produce
                an explainable preliminary assessment.
            </div>
            <div class="hero-badge">
                HACKATHON PROTOTYPE • NOT A SAFETY CERTIFICATION
            </div>
        </div>
        """
    )

    render_html(
        """
        <div class="section">
            <div class="section-title">How the system works</div>
            <div class="section-subtitle">
                A clean inspection pipeline from structural context to report.
            </div>
            <div class="step-row">
                <div class="step-card">
                    <div class="step-number">1</div>
                    <div class="step-title">Structural Context</div>
                    <div class="step-text">
                        Structure type, material, environment, applied load and
                        observed condition are recorded.
                    </div>
                </div>
                <div class="step-card">
                    <div class="step-number">2</div>
                    <div class="step-title">Computer Vision</div>
                    <div class="step-text">
                        OpenCV identifies crack-like visual regions and estimates
                        image-based dimensions.
                    </div>
                </div>
                <div class="step-card">
                    <div class="step-number">3</div>
                    <div class="step-title">AI + Risk Engine</div>
                    <div class="step-text">
                        AI vision findings and explainable rules provide
                        preliminary severity and possible causes.
                    </div>
                </div>
                <div class="step-card">
                    <div class="step-number">4</div>
                    <div class="step-title">Repair Guidance</div>
                    <div class="step-text">
                        Cause, solution, recommended repair and urgency are
                        included for detected damage.
                    </div>
                </div>
            </div>
        </div>
        """
    )

    render_html(
        """
        <div class="kpi-row">
            <div class="kpi"><div class="label">VISION</div><div class="value">OpenCV + AI</div></div>
            <div class="kpi"><div class="label">ENGINE</div><div class="value">Explainable</div></div>
            <div class="kpi"><div class="label">HISTORY</div><div class="value">SQLite</div></div>
            <div class="kpi"><div class="label">REPORT</div><div class="value">PDF</div></div>
        </div>
        """
    )

    render_html(
        """
        <div class="section">
            <div class="section-title">Engineering principle</div>
            <div class="section-subtitle">
                The prototype separates preliminary visual screening from
                professional structural diagnosis.
            </div>
            <div class="callout">
                <b>Important:</b> image pixels are not physical millimetres
                unless a reliable scale is supplied. Applied-load values are
                contextual inputs only and are never treated as proof of safety
                or load-bearing adequacy.
            </div>
        </div>
        """
    )


# ============================================================
# ANALYZE
# ============================================================
elif page == "Analyze":
    render_html(
        """
        <div class="hero">
            <div class="hero-kicker">01 / Inspection</div>
            <div class="hero-title">Analyze a Structure</div>
            <div class="hero-subtitle">
                Enter structural and loading context first, then upload a clear
                image for preliminary visual screening.
            </div>
        </div>
        """
    )

    render_html(
        """
        <div class="section">
            <div class="section-title">Structural information</div>
            <div class="section-subtitle">
                These fields provide engineering context for the screening rules.
            </div>
        </div>
        """
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        structure_type = st.selectbox(
            "Structure type",
            ["Building", "Bridge", "Retaining wall", "Industrial structure", "Other"],
        )
        building_type = st.selectbox(
            "Building type",
            ["Residential", "Commercial", "Industrial", "Institutional", "Other"],
        )
        element = st.selectbox(
            "Structural element",
            ["Wall", "Beam", "Column", "Slab", "Foundation", "Other / unknown"],
        )

    with c2:
        material = st.selectbox(
            "Primary material",
            [
                "Reinforced concrete",
                "Masonry / brick",
                "Concrete block",
                "Stone",
                "Steel",
                "Other",
            ],
        )
        location_on_structure = st.text_input(
            "Location on structure",
            placeholder="e.g. north wall, beam-column joint, slab edge",
        )
        age = st.text_input(
            "Approx. building age",
            value="10 years",
        )

    with c3:
        environment = st.selectbox(
            "Environment / exposure",
            [
                "Normal / dry",
                "Humid",
                "Coastal / marine",
                "Wet / seepage",
                "Industrial / chemical",
                "Outdoor exposed",
                "Unknown",
            ],
        )
        crack_type = st.selectbox(
            "Known / observed crack pattern",
            [
                "Auto / unknown",
                "Vertical",
                "Horizontal",
                "Diagonal",
                "Random / irregular",
                "Map-like",
            ],
        )
        scale = st.number_input(
            "Scale (mm per pixel, optional)",
            min_value=0.0,
            value=0.0,
            step=0.001,
            format="%.4f",
            help="Only enter this if you have a reliable physical reference in the image.",
        )

    render_html(
        """
        <div class="section">
            <div class="section-title">Applied load context</div>
            <div class="section-subtitle">
                Record the known or estimated applied load. This value is used
                only as context and does not prove structural safety.
            </div>
        </div>
        """
    )

    l1, l2, l3, l4 = st.columns(4)

    with l1:
        applied_load = st.number_input(
            "Applied load",
            min_value=0.0,
            value=0.0,
            step=0.1,
            format="%.3f",
            help="Enter 0 if unknown.",
        )

    with l2:
        load_unit = st.selectbox(
            "Load unit",
            ["kN", "N", "kg", "tonne", "kN/m", "kN/m²", "Unknown"],
        )

    with l3:
        load_type = st.selectbox(
            "Load type",
            [
                "Dead load",
                "Live load",
                "Point load",
                "Distributed load",
                "Impact load",
                "Equipment / machinery",
                "Unknown",
            ],
        )

    with l4:
        load_location = st.text_input(
            "Load location",
            placeholder="e.g. mid-span, floor area, beam end",
        )

    render_html(
        """
        <div class="warning">
            <b>Load-data safety note:</b>
            A manually entered load value is not a structural capacity calculation.
            Do not use this application to approve occupancy, remove supports,
            increase loads, or declare a structure safe.
        </div>
        """
    )

    render_html(
        """
        <div class="section">
            <div class="section-title">Condition indicators</div>
            <div class="section-subtitle">
                Visible or reported factors used by the explainable screening engine.
            </div>
        </div>
        """
    )

    q1, q2, q3 = st.columns(3)

    with q1:
        seepage = st.selectbox("Water seepage / dampness?", ["No", "Yes"])
        rust = st.selectbox("Visible rust / reinforcement?", ["No", "Yes"])

    with q2:
        progression = st.selectbox("Crack increasing?", ["No", "Yes"])
        event = st.selectbox("Recent earthquake / impact / event?", ["No", "Yes"])

    with q3:
        deformation = st.selectbox("Visible deformation / displacement?", ["No", "Yes"])
        sound = st.selectbox("Hollow / loose sound reported?", ["No", "Yes"])

    render_html(
        """
        <div class="section">
            <div class="section-title">Crack image</div>
            <div class="section-subtitle">
                Use a clear, focused image with good lighting whenever possible.
            </div>
        </div>
        """
    )

    uploaded = st.file_uploader(
        "Upload crack image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        left, right = st.columns(2)

        with left:
            st.image(image, caption="Uploaded image", width="stretch")

        with right:
            st.markdown("### Image preview")
            st.write(f"Resolution: **{image.width} × {image.height} px**")
            st.write("Analysis mode: **OpenCV + optional OpenAI Vision**")
            st.write(
                "AI status: **Available**"
                if get_api_key()
                else "AI status: **API key not configured — OpenCV still available**"
            )

        if st.button("🔎 Analyze Image", type="primary", width="stretch"):
            with st.spinner("Analyzing visible crack-like features..."):
                cv_result = cv_analyze(image)

                ctype = crack_type
                if crack_type == "Auto / unknown":
                    ctype = classify_crack(cv_result)

                info = {
                    "structure_type": structure_type,
                    "building_type": building_type,
                    "element": element,
                    "location_on_structure": location_on_structure,
                    "age": age,
                    "material": material,
                    "environment": environment,
                    "applied_load": applied_load,
                    "load_unit": load_unit,
                    "load_type": load_type,
                    "load_location": load_location,
                    "crack_type": ctype,
                    "seepage": seepage,
                    "rust": rust,
                    "progression": progression,
                    "event": event,
                    "deformation": deformation,
                    "sound": sound,
                }

                ai_result = analyze_image_with_ai(image, info)

                inputs = {
                    **info,
                    "width_mm": cv_result["width_px"] * scale if scale > 0 else 0,
                    "length_mm": cv_result["length_px"] * scale if scale > 0 else 0,
                    "detected": cv_result["detected"],
                    "confidence": cv_result["confidence"],
                    "length_px": cv_result["length_px"],
                    "width_px": cv_result["width_px"],
                    "area_px": cv_result["area_px"],
                    "image_width": image.width,
                    "image_height": image.height,
                    "ai_result": ai_result,
                    "ai_risk_score": ai_result.get("risk_score", 0),
                    "yolo_used": False,
                    "yolo_confidence": 0.0,
                }

                result = assess_risk(inputs)

                recommendation = build_recommendation(
                    inputs,
                    result,
                )

                recommendation.setdefault(
                    "headline",
                    "Preliminary screening completed.",
                )
                recommendation.setdefault(
                    "why",
                    ai_result.get("description", ""),
                )
                recommendation.setdefault(
                    "next_step",
                    ai_result.get("recommendation", ""),
                )
                recommendation.setdefault("safe_actions", [])

                length_text = format_measurement(
                    cv_result["length_px"],
                    scale,
                )
                width_text = format_measurement(
                    cv_result["width_px"],
                    scale,
                )

                st.session_state["last_analysis"] = {
                    "info": info,
                    "inputs": inputs,
                    "cv": cv_result,
                    "result": result,
                    "recommendation": recommendation,
                    "ai_result": ai_result,
                    "length_text": length_text,
                    "width_text": width_text,
                    "image": image,
                }

    analysis = st.session_state.get("last_analysis")

    if analysis:
        info = analysis["info"]
        cv_result = analysis["cv"]
        result = analysis["result"]
        recommendation = analysis["recommendation"]
        ai_result = analysis["ai_result"]
        length_text = analysis["length_text"]
        width_text = analysis["width_text"]
        report_image = analysis["image"]

        render_html(
            """
            <div class="section">
                <div class="section-title">Analysis result</div>
                <div class="section-subtitle">
                    Preliminary visual screening output. AI findings are visual
                    observations, not structural certification.
                </div>
            </div>
            """
        )

        severity = result.get(
            "severity",
            ai_result.get("severity", "Low"),
        )

        score = result.get(
            "score",
            result.get("risk_score", ai_result.get("risk_score", 0)),
        )

        box_class = (
            "danger"
            if severity == "Critical"
            else "warning"
            if severity in {"High", "Moderate"}
            else "good"
        )

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.metric("Severity", severity)
        with k2:
            st.metric("Screening score", score)
        with k3:
            st.metric("CV confidence", f'{cv_result.get("confidence", 0):.0f}%')
        with k4:
            st.metric("Crack type", info.get("crack_type", ""))

        render_html(
            f"""
            <div class="{box_class}">
                <b>{recommendation.get("headline", "")}</b>
                <br><br>
                {recommendation.get("why", "")}
            </div>
            """
        )

        render_html(
            """
            <div class="section">
                <div class="section-title">AI Visual Findings</div>
                <div class="section-subtitle">
                    Visible-image interpretation from OpenAI Vision. The model
                    does not determine hidden crack depth, structural capacity,
                    reinforcement condition or final structural cause.
                </div>
            </div>
            """
        )

        af1, af2 = st.columns([1, 1])

        with af1:
            render_html(
                f"""
                <div class="result-card">
                    <div class="result-title">Visual observation</div>
                    <p><b>Damage detected:</b> {"Yes" if ai_result.get("crack_detected") else "No / uncertain"}</p>
                    <p><b>Damage type:</b> {ai_result.get("damage_type", "")}</p>
                    <p><b>Visible location:</b> {ai_result.get("location", "")}</p>
                    <p><b>AI severity:</b> {ai_result.get("severity", "")}</p>
                    <p><b>AI risk score:</b> {ai_result.get("risk_score", 0)}/100</p>
                    <p><b>Description:</b> {ai_result.get("description", "")}</p>
                </div>
                """
            )

        with af2:
            render_html(
                """
                <div class="result-card">
                    <div class="result-title">Possible explanations</div>
                """
            )
            causes = ai_result.get("possible_causes", [])
            if causes:
                for cause in causes:
                    st.markdown(f"• {cause}")
            else:
                st.write("No reliable cause could be inferred from the image.")
            render_html("</div>")

        if ai_result.get("crack_detected") or cv_result.get("detected"):
            render_html(
                f"""
                <div class="section">
                    <div class="section-title">Cause & Recommended Repair</div>
                    <div class="section-subtitle">
                        Required four-part guidance for the detected crack/damage.
                    </div>
                </div>

                <div class="result-card">
                    <p><b>Cause</b><br>{ai_result.get("cause", "Undetermined from image")}</p>
                    <p><b>Solution</b><br>{ai_result.get("solution", "")}</p>
                    <p><b>Recommended Repair</b><br>{ai_result.get("recommended_repair", "")}</p>
                    <p><b>Urgency</b><br>{ai_result.get("urgency", "")}</p>
                </div>
                """
            )

        a, b = st.columns(2)

        with a:
            st.image(
                cv_result["overlay"],
                caption="OpenCV detected visual region",
                width="stretch",
            )

        with b:
            render_html(
                """
                <div class="result-card">
                    <div class="result-title">Measured visual evidence</div>
                """
            )
            st.write(f"**Estimated length:** {length_text}")
            st.write(f"**Estimated width:** {width_text}")
            st.write(
                f"**Visual extent:** {result.get('visual_evidence', 'Not available')}"
            )
            st.write("**Detection method:** OpenCV computer vision")
            render_html("</div>")

        render_html(
            """
            <div class="section">
                <div class="section-title">Recommended next step</div>
                <div class="section-subtitle">
                    Actions suggested by the preliminary screening.
                </div>
            </div>
            """
        )

        st.write(recommendation.get("next_step", ""))
        for item in recommendation.get("safe_actions", []):
            st.markdown(f"✓ {item}")

        render_html(
            """
            <div class="callout">
                <b>Important:</b> The application does not determine actual
                structural strength, remaining load capacity, reinforcement
                condition, crack depth or final structural cause. The entered
                applied load is contextual information only and is not evidence
                that the structure is safe.
            </div>
            """
        )

        pdf_bytes = make_pdf(
            info,
            result,
            recommendation,
            cv_result["confidence"],
            length_text,
            width_text,
            ai_result,
            report_image,
        )

        p1, p2 = st.columns(2)

        with p1:
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name="structure_doctor_report.pdf",
                mime="application/pdf",
                width="stretch",
            )

        with p2:
            if st.button("💾 Save to Inspection History", width="stretch"):
                save_inspection(
                    info,
                    result,
                    cv_result["confidence"],
                    length_text,
                    width_text,
                    recommendation,
                    ai_result,
                )
                st.success("Inspection saved successfully.")


# ============================================================
# TRACK CHANGE
# ============================================================
elif page == "Track Change":
    render_html(
        """
        <div class="hero">
            <div class="hero-kicker">02 / Monitoring</div>
            <div class="hero-title">Track Change</div>
            <div class="hero-subtitle">
                Compare two images using the same visual measurement pipeline.
            </div>
        </div>
        """
    )

    before = st.file_uploader(
        "Upload BEFORE image",
        type=["jpg", "jpeg", "png"],
        key="before",
    )

    after = st.file_uploader(
        "Upload AFTER image",
        type=["jpg", "jpeg", "png"],
        key="after",
    )

    if before and after:
        before_img = Image.open(before).convert("RGB")
        after_img = Image.open(after).convert("RGB")

        bcv = cv_analyze(before_img)
        acv = cv_analyze(after_img)

        c1, c2 = st.columns(2)

        with c1:
            st.image(before_img, caption="Before", width="stretch")
            st.metric("Before visible length", f'{bcv["length_px"]:.0f} px')

        with c2:
            st.image(after_img, caption="After", width="stretch")
            st.metric("After visible length", f'{acv["length_px"]:.0f} px')

        change = acv["length_px"] - bcv["length_px"]

        if change > 0:
            st.warning(
                "Visible crack-length proxy increased "
                f"by approximately {change:.0f} px."
            )
        elif change < 0:
            st.success(
                "Visible crack-length proxy decreased "
                f"by approximately {abs(change):.0f} px."
            )
        else:
            st.info("No measurable change in the image-based length proxy.")

        st.caption(
            "Image-based comparison only. Camera angle, distance, lighting "
            "and scale can affect the result."
        )


# ============================================================
# HISTORY
# ============================================================
elif page == "Inspection History":
    render_html(
        """
        <div class="hero">
            <div class="hero-kicker">03 / Records</div>
            <div class="hero-title">Inspection History</div>
            <div class="hero-subtitle">
                Review previously saved preliminary screening results.
            </div>
        </div>
        """
    )

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """
        SELECT
            timestamp,
            structure_type,
            building_type,
            element,
            location_on_structure,
            environment,
            applied_load,
            load_unit,
            load_type,
            load_location,
            crack_type,
            severity,
            score,
            confidence,
            length_value,
            width_value,
            cause,
            solution,
            recommended_repair,
            urgency
        FROM inspections
        ORDER BY id DESC
        """
    ).fetchall()
    conn.close()

    if not rows:
        st.info("No saved inspections yet.")
    else:
        for row in rows:
            (
                timestamp,
                structure_type,
                building_type,
                element,
                location_on_structure,
                environment,
                applied_load,
                load_unit,
                load_type,
                load_location,
                crack_type,
                severity,
                score,
                confidence,
                length_value,
                width_value,
                cause,
                solution,
                recommended_repair,
                urgency,
            ) = row

            with st.expander(
                f"{timestamp} • {element} • {severity}"
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Severity", severity)
                c2.metric("Score", score)
                c3.metric("Confidence", f"{confidence:.0f}%")
                c4.metric("Crack type", crack_type)

                st.write(
                    f"**Structure:** {structure_type} | "
                    f"**Building:** {building_type} | "
                    f"**Element:** {element}"
                )
                st.write(
                    f"**Location:** {location_on_structure or 'Not specified'} | "
                    f"**Environment:** {environment or 'Not specified'}"
                )
                st.write(
                    f"**Applied load:** {applied_load or 0} {load_unit or ''} | "
                    f"**Type:** {load_type or 'Unknown'} | "
                    f"**Location:** {load_location or 'Not specified'}"
                )
                st.write(
                    f"**Length:** {length_value} | **Width:** {width_value}"
                )

                if cause or solution or recommended_repair or urgency:
                    render_html(
                        f"""
                        <div class="result-card">
                            <div class="result-title">Cause & Recommended Repair</div>
                            <p><b>Cause:</b> {cause or "Undetermined"}</p>
                            <p><b>Solution:</b> {solution or "Not available"}</p>
                            <p><b>Recommended Repair:</b> {recommended_repair or "Not available"}</p>
                            <p><b>Urgency:</b> {urgency or "Not available"}</p>
                        </div>
                        """
                    )


# ============================================================
# FOOTER
# ============================================================
render_html(
    """
    <div class="footer">
        Structure Doctor AI • Hackathon Prototype
        <br>
        Preliminary image-based screening only • Not a structural safety certificate
    </div>
    """
)
