import io
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

from risk_engine import assess_risk, build_recommendation


APP_TITLE = "STRUCTURE DOCTOR AI"
DB_PATH = "inspection_history.db"


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Structure Doctor AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# PPT-style visual design
# -----------------------------
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

/* Hide default chrome */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {
    background: linear-gradient(
        135deg,
        #123B4A 0%,
        #1F6175 55%,
        #2F6F8F 100%
    );
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
    opacity: .82;
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
    max-width: 820px;
    opacity: .92;
}

.hero-badge {
    display: inline-block;
    margin-top: 22px;
    padding: 8px 13px;
    border: 1px solid rgba(255, 255, 255, .28);
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
}


/* =========================================================
   SECTION HEADERS
   ========================================================= */

.section {
    background: white;
    border: 1px solid #D9E2E8;
    border-radius: 18px;
    padding: 26px 28px;
    margin: 18px 0;
    box-shadow: 0 7px 22px rgba(31, 41, 51, .055);
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


/* =========================================================
   STREAMLIT WIDGET LABELS
   FIX FOR INVISIBLE / WHITE LABEL TEXT
   ========================================================= */

/* Main Streamlit widget label container */
[data-testid="stWidgetLabel"] {
    color: #123B4A !important;
    opacity: 1 !important;
}

/* Text inside widget labels */
[data-testid="stWidgetLabel"] p {
    color: #123B4A !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Selectbox labels */
.stSelectbox label {
    color: #123B4A !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Number input labels */
.stNumberInput label {
    color: #123B4A !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Text input labels */
.stTextInput label {
    color: #123B4A !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Text area labels */
.stTextArea label {
    color: #123B4A !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* File uploader labels */
.stFileUploader label {
    color: #123B4A !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Slider labels */
.stSlider label {
    color: #123B4A !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Radio labels outside sidebar */
.stRadio label {
    color: #123B4A !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Widget help / optional text */
[data-testid="stWidgetLabel"] small {
    color: #64748B !important;
    opacity: 1 !important;
}

/* Input description text */
[data-testid="InputInstructions"] {
    color: #64748B !important;
}


/* =========================================================
   INPUT CONTROLS
   ========================================================= */

.stSelectbox > div > div {
    border-radius: 10px;
}

.stNumberInput > div > div {
    border-radius: 10px;
}

.stTextInput > div > div {
    border-radius: 10px;
}

.stTextArea > div > div {
    border-radius: 10px;
}


/* =========================================================
   STEP CARDS
   ========================================================= */

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


/* =========================================================
   KPI
   ========================================================= */

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


/* =========================================================
   RESULT CARDS
   ========================================================= */

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


/* =========================================================
   CALLOUTS
   ========================================================= */

.callout,
.warning,
.danger,
.good {
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

.callout *,
.warning *,
.danger *,
.good * {
    color: #1F2933 !important;
}


/* =========================================================
   HERO TEXT OVERRIDE
   ========================================================= */

.hero,
.hero * {
    color: #FFFFFF !important;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background: #123B4A;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] .stRadio label {
    color: white !important;
    font-weight: 600;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {
    color: #1F2933 !important;
    border-radius: 10px;
    border: 1px solid #2F6F8F;
    font-weight: 700;
}

.stButton > button p {
    color: #1F2933 !important;
}


/* =========================================================
   METRICS
   ========================================================= */

[data-testid="stMetricLabel"] {
    color: #64748B !important;
}

[data-testid="stMetricValue"] {
    color: #1F2933 !important;
}


/* =========================================================
   CAPTIONS
   ========================================================= */

.stCaption,
[data-testid="stCaptionContainer"] {
    color: #64748B !important;
}


/* =========================================================
   PILLS
   ========================================================= */

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


/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;
    color: #718096;
    font-size: 11px;
    padding: 28px 10px 10px;
}


/* =========================================================
   RESPONSIVE
   ========================================================= */

@media (max-width: 850px) {
    .step-row,
    .kpi-row {
        grid-template-columns: repeat(2, 1fr);
    }

    .hero-title {
        font-size: 32px;
    }
}

@media (max-width: 560px) {
    .step-row,
    .kpi-row {
        grid-template-columns: 1fr;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Database
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            building_type TEXT,
            element TEXT,
            age TEXT,
            material TEXT,
            crack_type TEXT,
            severity TEXT,
            score INTEGER,
            confidence REAL,
            length_value TEXT,
            width_value TEXT,
            recommendation TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def save_inspection(
    info,
    result,
    confidence,
    length_text,
    width_text,
    recommendation,
):
    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
        INSERT INTO inspections
        (
            timestamp,
            building_type,
            element,
            age,
            material,
            crack_type,
            severity,
            score,
            confidence,
            length_value,
            width_value,
            recommendation
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            info.get("building_type", ""),
            info.get("element", ""),
            info.get("age", ""),
            info.get("material", ""),
            info.get("crack_type", ""),
            result.get("severity", ""),
            int(result.get("score", 0)),
            float(confidence),
            length_text,
            width_text,
            recommendation,
        ),
    )

    conn.commit()
    conn.close()


init_db()


# -----------------------------
# Image analysis
# -----------------------------
def cv_analyze(image: Image.Image):
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Dark linear features
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (17, 17),
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel,
    )

    _, mask = cv2.threshold(
        blackhat,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Edge information
    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    combined = cv2.bitwise_or(
        mask,
        edges,
    )

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5),
        ),
        iterations=1,
    )

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3),
        ),
        iterations=1,
    )

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            combined,
            8,
        )
    )

    h, w = gray.shape
    image_area = max(
        h * w,
        1,
    )

    candidates = []

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]

        if area < max(
            20,
            image_area * 0.00005,
        ):
            continue

        # Ignore huge background regions.
        if area > image_area * 0.35:
            continue

        aspect = max(
            bw,
            bh,
        ) / max(
            min(bw, bh),
            1,
        )

        if aspect < 2.0:
            continue

        candidates.append(
            (
                area,
                x,
                y,
                bw,
                bh,
                aspect,
            )
        )

    if not candidates:
        return {
            "detected": False,
            "confidence": 0.0,
            "length_px": 0.0,
            "width_px": 0.0,
            "area_px": 0.0,
            "overlay": image.convert("RGB"),
            "mask": Image.fromarray(
                np.zeros_like(gray)
            ),
            "yolo_used": False,
            "yolo_confidence": 0.0,
        }

    # Favor long, relatively thin regions.
    candidates.sort(
        key=lambda item:
            item[5] *
            np.sqrt(
                max(
                    item[0],
                    1,
                )
            ),
        reverse=True,
    )

    area, x, y, bw, bh, aspect = candidates[0]

    length_px = float(
        max(
            bw,
            bh,
        )
    )

    width_px = float(
        max(
            1.0,
            min(
                bw,
                bh,
            ) * 0.18,
        )
    )

    # Keep a reasonable visual estimate.
    width_px = min(
        width_px,
        max(
            length_px * 0.25,
            1.0,
        ),
    )

    confidence = min(
        98.0,
        max(
            55.0,
            55.0
            + min(
                aspect / 12.0,
                1.0,
            ) * 25.0
            + min(
                area / (image_area * 0.03),
                1.0,
            ) * 18.0,
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

    roi_mask[
        y:y + bh,
        x:x + bw
    ] = combined[
        y:y + bh,
        x:x + bw
    ]

    return {
        "detected": True,
        "confidence": float(confidence),
        "length_px": length_px,
        "width_px": width_px,
        "area_px": float(area),
        "overlay": Image.fromarray(overlay),
        "mask": Image.fromarray(roi_mask),
        "yolo_used": False,
        "yolo_confidence": 0.0,
    }


def format_measurement(value, scale):
    if scale and scale > 0:
        return f"{value * scale:.2f} mm"

    return f"{value:.0f} px"


def classify_crack(cv_result):
    length = cv_result.get(
        "length_px",
        0,
    )

    width = cv_result.get(
        "width_px",
        0,
    )

    if length <= 0:
        return "Auto / unknown"

    # Approximate visual classification from
    # dominant bounding-box dimensions.
    # This is intentionally framed as
    # a preliminary visual classification.
    if length > width * 7:
        return "Vertical / Horizontal"

    return "Random / irregular"


# -----------------------------
# PDF
# -----------------------------
def make_pdf(
    info,
    result,
    recommendation,
    confidence,
    length_text,
    width_text,
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

    body = ParagraphStyle(
        "Body2",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
    )

    story = [
        Paragraph(
            "STRUCTURE DOCTOR AI",
            title,
        ),
        Spacer(
            1,
            7 * mm,
        ),
        Paragraph(
            "AI-Assisted Crack Detection & Preliminary Structural Assessment",
            body,
        ),
        Spacer(
            1,
            5 * mm,
        ),
        Paragraph(
            "This report is a preliminary image-based screening result "
            "and is not a structural safety certificate.",
            body,
        ),
        Spacer(
            1,
            8 * mm,
        ),
    ]

    data = [
        [
            "Inspection",
            datetime.now().strftime(
                "%d %b %Y, %H:%M"
            ),
        ],
        [
            "Building type",
            str(
                info.get(
                    "building_type",
                    "",
                )
            ),
        ],
        [
            "Structural element",
            str(
                info.get(
                    "element",
                    "",
                )
            ),
        ],
        [
            "Building age",
            str(
                info.get(
                    "age",
                    "",
                )
            ),
        ],
        [
            "Material",
            str(
                info.get(
                    "material",
                    "",
                )
            ),
        ],
        [
            "Crack type",
            str(
                info.get(
                    "crack_type",
                    "",
                )
            ),
        ],
        [
            "Preliminary severity",
            result.get(
                "severity",
                "",
            ),
        ],
        [
            "Screening score",
            str(
                result.get(
                    "score",
                    0,
                )
            ),
        ],
        [
            "Confidence",
            f"{confidence:.0f}%",
        ],
        [
            "Estimated length",
            length_text,
        ],
        [
            "Estimated width",
            width_text,
        ],
    ]

    table = Table(
        data,
        colWidths=[
            55 * mm,
            115 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor("#EAF4F8"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#1F2933"),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, -1),
                    "Helvetica",
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#D9E2E8"),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(table)

    story.append(
        Spacer(
            1,
            7 * mm,
        )
    )

    story.append(
        Paragraph(
            "<b>What was observed</b>",
            body,
        )
    )

    story.append(
        Paragraph(
            result.get(
                "visual_evidence",
                "A crack-like visual region was identified "
                "for preliminary review.",
            ),
            body,
        )
    )

    story.append(
        Spacer(
            1,
            4 * mm,
        )
    )

    story.append(
        Paragraph(
            "<b>Possible explanations</b>",
            body,
        )
    )

    for cause in result.get(
        "possible_causes",
        [],
    ):
        story.append(
            Paragraph(
                f"• {cause}",
                body,
            )
        )

    story.append(
        Spacer(
            1,
            4 * mm,
        )
    )

    story.append(
        Paragraph(
            "<b>Recommended next step</b>",
            body,
        )
    )

    story.append(
        Paragraph(
            recommendation.get(
                "next_step",
                "",
            ),
            body,
        )
    )

    story.append(
        Spacer(
            1,
            4 * mm,
        )
    )

    story.append(
        Paragraph(
            "<b>Safety note</b>",
            body,
        )
    )

    story.append(
        Paragraph(
            "The system does not determine actual structural strength, "
            "remaining load capacity, reinforcement condition, or final "
            "structural cause. A qualified structural professional should "
            "assess significant or uncertain damage.",
            body,
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 🏗️ Structure Doctor")
    st.caption("AI-assisted preliminary screening")

    page = st.radio(
        "Navigate",
        [
            "Home",
            "Analyze",
            "Track Change",
            "Inspection History",
        ],
    )

    st.markdown("---")

    st.markdown("### System")

    st.markdown(
        '<span class="pill">OpenCV</span>'
        '<span class="pill">Python</span>'
        '<span class="pill">SQLite</span>',
        unsafe_allow_html=True,
    )


# -----------------------------
# Home
# -----------------------------
if page == "Home":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">
                AI-Assisted Structural Screening
            </div>

            <div class="hero-title">
                STRUCTURE<br>DOCTOR AI
            </div>

            <div class="hero-subtitle">
                A visual screening prototype that analyzes uploaded crack images,
                extracts measurable image features, combines them with structural
                context, and produces an explainable preliminary assessment.
            </div>

            <div class="hero-badge">
                HACKATHON PROTOTYPE • NOT A SAFETY CERTIFICATION
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section">
            <div class="section-title">
                How the system works
            </div>

            <div class="section-subtitle">
                A presentation-style pipeline from image input
                to explainable recommendation.
            </div>

            <div class="step-row">

                <div class="step-card">
                    <div class="step-number">1</div>
                    <div class="step-title">Input</div>
                    <div class="step-text">
                        Structural details, visible condition indicators
                        and a crack image.
                    </div>
                </div>

                <div class="step-card">
                    <div class="step-number">2</div>
                    <div class="step-title">Computer Vision</div>
                    <div class="step-text">
                        OpenCV identifies crack-like regions and estimates
                        image-based dimensions.
                    </div>
                </div>

                <div class="step-card">
                    <div class="step-number">3</div>
                    <div class="step-title">Risk Engine</div>
                    <div class="step-text">
                        Explainable rules combine visual evidence
                        with structural context.
                    </div>
                </div>

                <div class="step-card">
                    <div class="step-number">4</div>
                    <div class="step-title">Recommendation</div>
                    <div class="step-text">
                        The system provides severity, possible explanations
                        and next steps.
                    </div>
                </div>

            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="kpi-row">

            <div class="kpi">
                <div class="label">VISION</div>
                <div class="value">OpenCV</div>
            </div>

            <div class="kpi">
                <div class="label">ENGINE</div>
                <div class="value">Explainable</div>
            </div>

            <div class="kpi">
                <div class="label">HISTORY</div>
                <div class="value">SQLite</div>
            </div>

            <div class="kpi">
                <div class="label">REPORT</div>
                <div class="value">PDF</div>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section">

            <div class="section-title">
                Engineering principle
            </div>

            <div class="section-subtitle">
                The prototype deliberately separates visual screening
                from professional structural diagnosis.
            </div>

            <div class="callout">
                <b>Important:</b>
                Image pixels are not physical millimetres unless a scale
                is supplied. The system therefore uses normalized visual
                features when no physical reference is available.
                It does not claim to calculate actual structural strength
                or load-bearing capacity.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Analyze
# -----------------------------
elif page == "Analyze":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-kicker">
                01 / Inspection
            </div>

            <div class="hero-title">
                Analyze a Structure
            </div>

            <div class="hero-subtitle">
                Enter the structural context first, then upload
                the image for visual screening.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section">

            <div class="section-title">
                Structural information
            </div>

            <div class="section-subtitle">
                These inputs provide engineering context for
                the screening rules.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        building_type = st.selectbox(
            "Building type",
            [
                "Residential",
                "Commercial",
                "Industrial",
                "Institutional",
                "Other",
            ],
        )

        element = st.selectbox(
            "Structural element",
            [
                "Wall",
                "Beam",
                "Column",
                "Slab",
                "Foundation",
                "Other / unknown",
            ],
        )

    with c2:

        age = st.text_input(
            "Approx. building age",
            value="10 years",
        )

        material = st.selectbox(
            "Material",
            [
                "Reinforced concrete",
                "Masonry / brick",
                "Concrete block",
                "Stone",
                "Other",
            ],
        )

    with c3:

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
            help="Only enter this if you know a reliable physical reference.",
        )

    st.markdown(
        """
        <div class="section">

            <div class="section-title">
                Condition indicators
            </div>

            <div class="section-subtitle">
                Visible/contextual factors used by the explainable
                screening engine.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    q1, q2, q3 = st.columns(3)

    with q1:

        seepage = st.selectbox(
            "Water seepage / dampness?",
            [
                "No",
                "Yes",
            ],
        )

        rust = st.selectbox(
            "Visible rust / reinforcement?",
            [
                "No",
                "Yes",
            ],
        )

    with q2:

        progression = st.selectbox(
            "Crack increasing?",
            [
                "No",
                "Yes",
            ],
        )

        event = st.selectbox(
            "Recent earthquake / impact / event?",
            [
                "No",
                "Yes",
            ],
        )

    with q3:

        deformation = st.selectbox(
            "Visible deformation / displacement?",
            [
                "No",
                "Yes",
            ],
        )

        sound = st.selectbox(
            "Hollow / loose sound reported?",
            [
                "No",
                "Yes",
            ],
        )

    st.markdown(
        """
        <div class="section">

            <div class="section-title">
                Crack image
            </div>

            <div class="section-subtitle">
                Use a clear, focused image with good lighting whenever possible.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload crack image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
    )

    if uploaded:

        image = Image.open(
            uploaded
        ).convert("RGB")

        left, right = st.columns(2)

        with left:

            st.image(
                image,
                caption="Uploaded image",
                width="stretch",
            )

        with right:

            st.markdown("### Image preview")

            st.write(
                f"Resolution: **{image.width} × {image.height} px**"
            )

            st.write(
                "Analysis mode: **OpenCV computer vision**"
            )

            st.write(
                "Model status: **No external model required**"
            )

        if st.button(
            "🔎 Analyze Image",
            type="primary",
            width="stretch",
        ):

            with st.spinner(
                "Analyzing visible crack-like features..."
            ):

                cv = cv_analyze(
                    image
                )

            ctype = crack_type

            if crack_type == "Auto / unknown":
                ctype = classify_crack(
                    cv
                )

            info = {
                "building_type": building_type,
                "element": element,
                "age": age,
                "material": material,
                "crack_type": ctype,
                "seepage": seepage,
                "rust": rust,
                "progression": progression,
                "event": event,
                "deformation": deformation,
                "sound": sound,
            }

            inputs = {
                **info,
                "width_mm": (
                    cv["width_px"] * scale
                    if scale > 0
                    else 0
                ),
                "length_mm": (
                    cv["length_px"] * scale
                    if scale > 0
                    else 0
                ),
                "detected": cv["detected"],
                "confidence": cv["confidence"],
                "length_px": cv["length_px"],
                "width_px": cv["width_px"],
                "area_px": cv["area_px"],
                "image_width": image.width,
                "image_height": image.height,
                "yolo_used": False,
                "yolo_confidence": 0.0,
            }

            result = assess_risk(
                inputs
            )

            recommendation = build_recommendation(
                inputs,
                result,
            )

            length_text = format_measurement(
                cv["length_px"],
                scale,
            )

            width_text = format_measurement(
                cv["width_px"],
                scale,
            )

            st.session_state["last_analysis"] = {
                "info": info,
                "inputs": inputs,
                "cv": cv,
                "result": result,
                "recommendation": recommendation,
                "length_text": length_text,
                "width_text": width_text,
            }

    analysis = st.session_state.get(
        "last_analysis"
    )

    if analysis:

        info = analysis["info"]
        cv = analysis["cv"]
        result = analysis["result"]
        recommendation = analysis["recommendation"]
        length_text = analysis["length_text"]
        width_text = analysis["width_text"]

        st.markdown(
            """
            <div class="section">

                <div class="section-title">
                    Analysis result
                </div>

                <div class="section-subtitle">
                    Preliminary visual screening output.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        severity = result.get(
            "severity",
            "Low",
        )

        box_class = (
            "danger"
            if severity == "Critical"
            else "warning"
            if severity in {
                "High",
                "Moderate",
            }
            else "good"
        )

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.metric(
                "Severity",
                severity,
            )

        with k2:
            st.metric(
                "Score",
                result.get(
                    "score",
                    0,
                ),
            )

        with k3:
            st.metric(
                "Confidence",
                f'{cv.get("confidence", 0):.0f}%',
            )

        with k4:
            st.metric(
                "Crack type",
                info.get(
                    "crack_type",
                    "",
                ),
            )

        st.markdown(
            f"""
            <div class="{box_class}">
                <b>{recommendation.get("headline", "")}</b>
                <br><br>
                {recommendation.get("why", "")}
            </div>
            """,
            unsafe_allow_html=True,
        )

        a, b = st.columns(2)

        with a:

            st.image(
                cv["overlay"],
                caption="Detected visual region",
                width="stretch",
            )

        with b:

            st.markdown(
                """
                <div class="result-card">

                    <div class="result-title">
                        Measured visual evidence
                    </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(
                f"**Estimated length:** {length_text}"
            )

            st.write(
                f"**Estimated width:** {width_text}"
            )

            st.write(
                f"**Visual extent:** "
                f"{result.get('visual_evidence', 'Not available')}"
            )

            st.write(
                "**Detection method:** OpenCV computer vision"
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        r1, r2 = st.columns(2)

        with r1:

            st.markdown(
                """
                <div class="result-card">

                    <div class="result-title">
                        Possible explanations
                    </div>
                """,
                unsafe_allow_html=True,
            )

            for cause in result.get(
                "possible_causes",
                [],
            ):
                st.markdown(
                    f"• {cause}"
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        with r2:

            st.markdown(
                """
                <div class="result-card">

                    <div class="result-title">
                        Recommended next step
                    </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(
                recommendation.get(
                    "next_step",
                    "",
                )
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="section">

                <div class="section-title">
                    Safe actions
                </div>

                <div class="section-subtitle">
                    Actions suggested by the screening result.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        for item in recommendation.get(
            "safe_actions",
            [],
        ):
            st.markdown(
                f"✓ {item}"
            )

        st.info(
            "Important: This prototype does not determine actual "
            "structural strength, remaining load capacity, "
            "reinforcement condition, or final structural cause."
        )

        pdf_bytes = make_pdf(
            info,
            result,
            recommendation,
            cv["confidence"],
            length_text,
            width_text,
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

            if st.button(
                "💾 Save to Inspection History",
                width="stretch",
            ):

                save_inspection(
                    info,
                    result,
                    cv["confidence"],
                    length_text,
                    width_text,
                    recommendation.get(
                        "next_step",
                        "",
                    ),
                )

                st.success(
                    "Inspection saved successfully."
                )


# -----------------------------
# Track change
# -----------------------------
elif page == "Track Change":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-kicker">
                02 / Monitoring
            </div>

            <div class="hero-title">
                Track Change
            </div>

            <div class="hero-subtitle">
                Compare two images using the same visual
                measurement pipeline.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    before = st.file_uploader(
        "Upload BEFORE image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        key="before",
    )

    after = st.file_uploader(
        "Upload AFTER image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        key="after",
    )

    if before and after:

        before_img = Image.open(
            before
        ).convert("RGB")

        after_img = Image.open(
            after
        ).convert("RGB")

        bcv = cv_analyze(
            before_img
        )

        acv = cv_analyze(
            after_img
        )

        c1, c2 = st.columns(2)

        with c1:

            st.image(
                before_img,
                caption="Before",
                width="stretch",
            )

            st.metric(
                "Before visible length",
                f'{bcv["length_px"]:.0f} px',
            )

        with c2:

            st.image(
                after_img,
                caption="After",
                width="stretch",
            )

            st.metric(
                "After visible length",
                f'{acv["length_px"]:.0f} px',
            )

        change = (
            acv["length_px"]
            - bcv["length_px"]
        )

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

            st.info(
                "No measurable change in the image-based "
                "length proxy."
            )

        st.caption(
            "This is an image-based comparison only. "
            "Different camera angle, distance, lighting "
            "and scale can affect the result."
        )


# -----------------------------
# History
# -----------------------------
elif page == "Inspection History":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-kicker">
                03 / Records
            </div>

            <div class="hero-title">
                Inspection History
            </div>

            <div class="hero-subtitle">
                Review previously saved preliminary
                screening results.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    conn = sqlite3.connect(
        DB_PATH
    )

    rows = conn.execute(
        """
        SELECT
            timestamp,
            building_type,
            element,
            crack_type,
            severity,
            score,
            confidence,
            length_value,
            width_value
        FROM inspections
        ORDER BY id DESC
        """
    ).fetchall()

    conn.close()

    if not rows:

        st.info(
            "No saved inspections yet."
        )

    else:

        for row in rows:

            (
                timestamp,
                btype,
                element,
                crack_type,
                severity,
                score,
                confidence,
                length_value,
                width_value,
            ) = row

            with st.expander(
                f"{timestamp} • {element} • {severity}"
            ):

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Severity",
                    severity,
                )

                c2.metric(
                    "Score",
                    score,
                )

                c3.metric(
                    "Confidence",
                    f"{confidence:.0f}%",
                )

                c4.metric(
                    "Crack type",
                    crack_type,
                )

                st.write(
                    f"**Building:** {btype}  |  "
                    f"**Element:** {element}  |  "
                    f"**Length:** {length_value}  |  "
                    f"**Width:** {width_value}"
                )


# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <div class="footer">
        Structure Doctor AI • Hackathon Prototype
        <br>
        Preliminary image-based screening only •
        Not a structural safety certificate
    </div>
    """,
    unsafe_allow_html=True,
)
