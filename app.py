import io
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

from risk_engine import assess_risk, build_recommendation


APP_TITLE = "STRUCTURE DOCTOR"
DB_PATH = Path("structure_doctor.db")


# -----------------------------
# Page / theme
# -----------------------------
st.set_page_config(
    page_title="Structure Doctor AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif;
}
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 2.3rem 2.5rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #101820 0%, #1d3542 60%, #2f6f8f 100%);
    color: white;
    margin-bottom: 1.4rem;
}
.hero h1 {
    font-size: 3.3rem;
    margin: 0;
    letter-spacing: -2px;
}
.hero .tagline {
    font-size: 1.15rem;
    margin-top: .7rem;
    opacity: .92;
}
.hero .tech {
    margin-top: 1.2rem;
    font-size: .82rem;
    letter-spacing: 1.5px;
    font-weight: 700;
    opacity: .75;
}
.section {
    border: 1px solid #d9e2e8;
    background: #fff;
    border-radius: 18px;
    padding: 1.2rem 1.35rem;
    margin-bottom: 1rem;
}
.kpi {
    border: 1px solid #d9e2e8;
    border-radius: 16px;
    padding: 1rem;
    background: #f7f9fb;
    min-height: 105px;
}
.kpi .label {
    color: #65737e;
    font-size: .8rem;
    text-transform: uppercase;
    letter-spacing: .7px;
}
.kpi .value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    margin-top: .35rem;
    color: #1F2933 !important;
}
.badge {
    display: inline-block;
    border-radius: 999px;
    padding: .35rem .8rem;
    font-weight: 700;
    font-size: .8rem;
    background: #e8f1f5;
}
.callout {
    padding: 1rem 1.1rem;
    border-radius: 14px;
    background: #f7f9fb;
    border-left: 5px solid #2f6f8f;
    color: #1F2933 !important;
}

.warning {
    padding: 1rem 1.1rem;
    border-radius: 14px;
    background: #fff6e8;
    border-left: 5px solid #d99121;
    color: #1F2933 !important;
}

.danger {
    padding: 1rem 1.1rem;
    border-radius: 14px;
    background: #fff0f0;
    border-left: 5px solid #c53d3d;
    color: #1F2933 !important;
}

.good {
    padding: 1rem 1.1rem;
    border-radius: 14px;
    background: #edf8f1;
    border-left: 5px solid #3f8f5f;
    color: #1F2933 !important;
}
.small {
    color: #65737e;
    font-size: .86rem;
}
.step {
    text-align: center;
    padding: 1rem .5rem;
}
.step-number {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #e8f1f5;
    color: #2f6f8f;
    font-weight: 700;
    margin-bottom: .4rem;
}
hr { border: none; border-top: 1px solid #e1e7eb; }
.kpi .value,
.callout *,
.warning *,
.danger *,
.good * {
    color: #1F2933 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Database
# -----------------------------
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            building_type TEXT,
            element TEXT,
            age REAL,
            material TEXT,
            crack_type TEXT,
            severity TEXT,
            confidence REAL,
            length_text TEXT,
            width_text TEXT,
            recommendation TEXT
        )
        """
    )
    con.commit()
    con.close()


def save_inspection(result, info):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """
        INSERT INTO inspections
        (created_at, building_type, element, age, material, crack_type,
         severity, confidence, length_text, width_text, recommendation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            info["building_type"],
            info["element"],
            info["age"],
            info["material"],
            result["crack_type"],
            result["severity"],
            result["confidence"],
            result["length_text"],
            result["width_text"],
            result["recommendation"]["next_step"],
        ),
    )
    con.commit()
    con.close()


init_db()


# -----------------------------
# Image analysis
# -----------------------------
def cv_analyze(uploaded_file, scale_mm_per_pixel=0.0):
    image = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Dark-line / edge enhancement.
    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, blackhat_kernel)

    _, mask = cv2.threshold(
        blackhat, max(8, int(np.percentile(blackhat, 88))), 255, cv2.THRESH_BINARY
    )

    # Also retain strong Canny edges to make the demo more tolerant of different images.
    edges = cv2.Canny(gray, 50, 150)
    combined = cv2.bitwise_or(mask, edges)

    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # Remove tiny noise.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(combined, 8)
    clean = np.zeros_like(combined)
    candidates = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= max(20, int(rgb.shape[0] * rgb.shape[1] * 0.00005)):
            candidates.append((area, i))
    candidates.sort(reverse=True)

    for area, i in candidates[:12]:
        clean[labels == i] = 255

    # A crack is usually thin. Keep a conservative region and derive a visual score.
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        overlay = image.copy()
        return {
            "image": image,
            "overlay": overlay,
            "detected": False,
            "confidence": 28,
            "length_px": 0.0,
            "width_px": 0.0,
            "area_px": 0.0,
            "start": (0, 0),
            "end": (0, 0),
            "mask": clean,
        }

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contour = contours[0]
    area_px = float(cv2.contourArea(contour))
    x, y, w, h = cv2.boundingRect(contour)

    # Estimate a path-like length from the bounding-box diagonal/perimeter.
    length_px = max(float(np.hypot(w, h)), float(cv2.arcLength(contour, False)) * 0.45)
    length_px = min(length_px, float(max(rgb.shape[:2]) * 2.5))

    # Area / length gives a rough visible width proxy.
    width_px = max(1.0, min(80.0, (area_px / max(length_px, 1.0)) * 1.7))

    # Endpoints are approximated from farthest contour points.
    pts = contour.reshape(-1, 2)
    if len(pts) >= 2:
        # Fast approximation: extremes along y/x.
        p1 = pts[np.argmin(pts[:, 0] + pts[:, 1])]
        p2 = pts[np.argmax(pts[:, 0] + pts[:, 1])]
        start = (int(p1[0]), int(p1[1]))
        end = (int(p2[0]), int(p2[1]))
    else:
        start, end = (x, y), (x + w, y + h)

    # Heuristic confidence: stronger, thinner, longer candidate -> higher.
    density = min(1.0, area_px / max(rgb.shape[0] * rgb.shape[1] * 0.002, 1))
    thinness = max(0.0, 1.0 - min(width_px / 25.0, 1.0))
    extent = min(1.0, length_px / max(rgb.shape[0], rgb.shape[1]))
    confidence = int(np.clip(45 + 28 * density + 17 * thinness + 10 * extent, 35, 94))

    overlay_bgr = bgr.copy()
    overlay_bgr[clean > 0] = (0, 0, 255)
    overlay_bgr = cv2.addWeighted(bgr, 0.72, overlay_bgr, 0.28, 0)

    cv2.circle(overlay_bgr, start, 7, (0, 220, 0), -1)
    cv2.circle(overlay_bgr, end, 7, (255, 150, 0), -1)
    cv2.putText(
        overlay_bgr,
        "AI-assisted crack region",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    overlay = Image.fromarray(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))

    return {
        "image": image,
        "overlay": overlay,
        "detected": confidence >= 48,
        "confidence": confidence,
        "length_px": length_px,
        "width_px": width_px,
        "area_px": area_px,
        "start": start,
        "end": end,
        "mask": clean,
    }


def format_measurement(px, scale, unit="mm"):
    if px <= 0:
        return "Not clearly measurable"
    if scale and scale > 0:
        return f"{px * scale:.2f} {unit}"
    return f"{px:.0f} px (no scale provided)"


def classify_crack(user_type, cv):
    if user_type != "Auto / unknown":
        return user_type
    if not cv["detected"]:
        return "Not clearly detected"

    sx, sy = cv["start"]
    ex, ey = cv["end"]
    dx, dy = abs(ex - sx), abs(ey - sy)

    if dy > dx * 2:
        return "Vertical"
    if dx > dy * 2:
        return "Horizontal"
    if dx > 20 and dy > 20:
        return "Diagonal"
    return "Random / irregular"


# -----------------------------
# PDF report
# -----------------------------
def make_pdf(result, info, recommendation):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Title2",
            parent=styles["Title"],
            fontSize=24,
            leading=28,
            textColor=colors.HexColor("#1D3542"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small2",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#56636C"),
        )
    )

    story = [
        Paragraph("STRUCTURE DOCTOR", styles["Title2"]),
        Paragraph("AI-Assisted Crack Detection & Preliminary Assessment", styles["Heading2"]),
        Spacer(1, 5 * mm),
        Paragraph(
            f"Inspection timestamp: {datetime.now().strftime('%d %b %Y, %H:%M')}",
            styles["Small2"],
        ),
        Spacer(1, 5 * mm),
    ]

    data = [
        ["Building type", info["building_type"]],
        ["Structural element", info["element"]],
        ["Approx. building age", f'{info["age"]:.0f} years'],
        ["Material", info["material"]],
        ["Crack type", result["crack_type"]],
        ["Preliminary severity", result["severity"]],
        ["Confidence", f'{result["confidence"]}%'],
        ["Estimated length", result["length_text"]],
        ["Estimated width", result["width_text"]],
    ]
    table = Table(data, colWidths=[52 * mm, 118 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F7")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9E2E8")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story += [table, Spacer(1, 6 * mm)]

    story.append(Paragraph("What was observed", styles["Heading2"]))
    story.append(Paragraph(result["observation"], styles["BodyText"]))
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("Possible explanations", styles["Heading2"]))
    for item in result["possible_causes"]:
        story.append(Paragraph("• " + item, styles["BodyText"]))
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("Recommended next step", styles["Heading2"]))
    story.append(Paragraph(recommendation["next_step"], styles["BodyText"]))
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("Safe immediate actions", styles["Heading2"]))
    for item in recommendation["safe_actions"]:
        story.append(Paragraph("• " + item, styles["BodyText"]))
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("Possible maintenance / remedy category", styles["Heading2"]))
    story.append(Paragraph(recommendation["remedy"], styles["BodyText"]))
    story.append(Spacer(1, 5 * mm))

    disclaimer = (
        "<b>IMPORTANT:</b> This report is a preliminary visual screening output. "
        "It does not determine actual structural strength, load-bearing capacity, "
        "structural safety, engineering diagnosis, design, repair specification or certification. "
        "Possible causes and remedies are not confirmed from an image alone. "
        "A qualified professional should inspect potentially significant damage."
    )
    story.append(Paragraph(disclaimer, styles["Small2"]))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# -----------------------------
# Sidebar / landing
# -----------------------------
with st.sidebar:
    st.markdown("## 🏗️ Structure Doctor")
    st.caption("See the Crack. Understand the Risk. Act Early.")
    page = st.radio(
        "Navigate",
        ["Home", "Analyze", "Track Change", "Inspection History"],
        index=0,
    )
    st.divider()
    st.markdown("### Prototype status")
    st.caption("AI-assisted / computer-vision demo")
    st.caption("Preliminary visual screening only.")
    st.caption("Not an engineering certification tool.")


def home():
    st.markdown(
        """
<div class="hero">
  <div class="tech">AI + COMPUTER VISION + CIVIL ENGINEERING</div>
  <h1>STRUCTURE<br>DOCTOR</h1>
  <div class="tagline">See the Crack. Understand the Risk. Act Early.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### From a simple photo to clearer crack understanding")
    cols = st.columns(5)
    steps = [
        ("01", "Upload", "Crack image"),
        ("02", "Detect", "AI-assisted region"),
        ("03", "Measure", "Length / width"),
        ("04", "Assess", "Type / severity"),
        ("05", "Act", "Guidance / report"),
    ]
    for c, (n, title, sub) in zip(cols, steps):
        with c:
            st.markdown(
                f"""
<div class="step">
<div class="step-number">{n}</div>
<b>{title}</b><br><span class="small">{sub}</span>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Why Structure Doctor?")
        st.write(
            "Small cracks are easy to overlook. The prototype helps users document "
            "visible cracks, understand preliminary indicators, compare changes over time, "
            "and decide when professional inspection should be sought."
        )
    with c2:
        st.markdown("### What it can show")
        st.write(
            "Crack detection overlay • estimated measurements • classification • "
            "preliminary severity • possible causes • safe immediate actions • "
            "possible maintenance category • inspection recommendation • PDF report."
        )

    st.markdown(
        """
<div class="warning">
<b>Safety boundary:</b> A photo cannot confirm structural capacity or certify a repair.
Structure Doctor is a preliminary visual screening prototype. Significant, uncertain,
rapidly changing, or potentially structural damage should be professionally inspected.
</div>
""",
        unsafe_allow_html=True,
    )


def analyze_page():
    st.title("🔎 Crack Analysis")
    st.caption("Enter structural context first, then upload the crack image.")

    st.markdown("### 01 — Structural information")
    c1, c2, c3 = st.columns(3)
    with c1:
        building_type = st.selectbox(
            "Building / structure type",
            ["Residential", "Commercial", "Institutional", "Industrial", "Bridge / infrastructure", "Other"],
        )
    with c2:
        element = st.selectbox(
            "Structural element",
            ["Wall", "Slab", "Beam", "Column", "Foundation", "Masonry", "Other / unknown"],
        )
    with c3:
        age = st.number_input("Approx. building age (years)", 0, 200, 10)

    c1, c2, c3 = st.columns(3)
    with c1:
        material = st.selectbox(
            "Material",
            ["Reinforced concrete", "Masonry / brick", "Plaster / finish", "Steel", "Other / unknown"],
        )
    with c2:
        crack_type_input = st.selectbox(
            "Known crack pattern",
            ["Auto / unknown", "Vertical", "Horizontal", "Diagonal", "Random / irregular", "Map-like"],
        )
    with c3:
        scale = st.number_input(
            "Optional scale (mm per pixel)",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=0.01,
            help="Only enter this if you have a reliable image scale/reference. Otherwise measurements remain pixel estimates.",
        )

    st.markdown("### 02 — Visible condition indicators")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        seepage = st.selectbox("Water seepage / dampness", ["Unknown", "No", "Yes"])
    with c2:
        rust = st.selectbox("Visible rust / reinforcement", ["Unknown", "No", "Yes"])
    with c3:
        progression = st.selectbox("Crack increasing?", ["Unknown", "No", "Yes"])
    with c4:
        event = st.selectbox("Recent earthquake / impact / unusual event", ["Unknown", "No", "Yes"])

    c1, c2 = st.columns(2)
    with c1:
        deformation = st.selectbox("Visible deformation / displacement", ["Unknown", "No", "Yes"])
    with c2:
        sound = st.selectbox("Unusual hollow sound / loose area", ["Unknown", "No", "Yes"])

    st.markdown("### 03 — Crack image")
    uploaded = st.file_uploader(
        "Upload JPG, JPEG or PNG",
        type=["jpg", "jpeg", "png"],
        help="For the best comparison, use a clear image with stable lighting and a reference scale if available.",
    )

    if uploaded is None:
        st.info("Upload an image to start the analysis.")
        return

    if st.button("🚀 Analyze Crack", type="primary", use_container_width=True):
        with st.spinner("Preprocessing image and running AI-assisted visual screening..."):
            cv = cv_analyze(uploaded, scale)

            info = {
                "building_type": building_type,
                "element": element,
                "age": float(age),
                "material": material,
                "seepage": seepage,
                "rust": rust,
                "progression": progression,
                "event": event,
                "deformation": deformation,
                "sound": sound,
            }

            ctype = classify_crack(crack_type_input, cv)
            inputs = {
                **info,
                "crack_type": ctype,
                "width_mm": cv["width_px"] * scale if scale > 0 else 0,
                "length_mm": cv["length_px"] * scale if scale > 0 else 0,
                "detected": cv["detected"],
                "confidence": cv["confidence"],
            }

            result = assess_risk(inputs)
            result["crack_type"] = ctype
            result["confidence"] = cv["confidence"]
            result["length_text"] = format_measurement(cv["length_px"], scale)
            result["width_text"] = format_measurement(cv["width_px"], scale)
            result["start"] = cv["start"]
            result["end"] = cv["end"]
            result["observation"] = (
                "A crack-like visual region was detected and highlighted for preliminary review."
                if cv["detected"]
                else "No sufficiently clear crack-like region was confidently detected."
            )
            result["possible_causes"] = result["possible_causes"]

            recommendation = build_recommendation(inputs, result)
            result["recommendation"] = recommendation

            st.session_state["analysis"] = {
                "cv": cv,
                "result": result,
                "info": info,
                "recommendation": recommendation,
            }

    analysis = st.session_state.get("analysis")
    if not analysis:
        return

    cv = analysis["cv"]
    result = analysis["result"]
    info = analysis["info"]
    recommendation = analysis["recommendation"]

    st.markdown("---")
    st.markdown("## 04 — See what the AI sees")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**ORIGINAL IMAGE**")
        st.image(cv["image"], use_container_width=True)
    with c2:
        st.markdown("**AI-ASSISTED DETECTION OVERLAY**")
        st.image(cv["overlay"], use_container_width=True)

    k1, k2, k3, k4 = st.columns(4)
    for col, label, value in [
        (k1, "Detection", "Crack detected" if cv["detected"] else "Not clear"),
        (k2, "Confidence", f'{result["confidence"]}%'),
        (k3, "Length", result["length_text"]),
        (k4, "Avg. width", result["width_text"]),
    ]:
        with col:
            st.markdown(
                f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Detection details")
    d1, d2, d3 = st.columns(3)
    d1.metric("Crack type", result["crack_type"])
    d2.metric("Start point", f'X {result["start"][0]} / Y {result["start"][1]}')
    d3.metric("End point", f'X {result["end"][0]} / Y {result["end"][1]}')

    st.caption(
        "DEMO / AI-ASSISTED VISUAL SCREENING — measurements are estimates unless a reliable image scale is supplied."
    )

    st.markdown("---")
    st.markdown("## 05 — Classification, severity & possible causes")
    s1, s2 = st.columns([1, 2])
    with s1:
        st.markdown(
            f'<div class="kpi"><div class="label">Preliminary severity</div><div class="value">{result["severity"]}</div></div>',
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown("**Possible explanations — not confirmed from image alone**")
        for cause in result["possible_causes"]:
            st.write("• " + cause)

    st.markdown("---")
    st.markdown("## 06 — Recommendation & possible remedy")

    if recommendation["level"] == "CRITICAL":
        box_class = "danger"
    elif recommendation["level"] == "HIGH":
        box_class = "warning"
    elif recommendation["level"] == "MODERATE":
        box_class = "warning"
    else:
        box_class = "good"

    st.markdown(
        f'<div class="{box_class}"><b>{recommendation["headline"]}</b><br>{recommendation["next_step"]}</div>',
        unsafe_allow_html=True,
    )

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("### What you can do now")
        for item in recommendation["safe_actions"]:
            st.write("✓ " + item)
    with r2:
        st.markdown("### Possible maintenance / remedy")
        st.write(recommendation["remedy"])

    st.markdown("### Why this recommendation?")
    st.write(recommendation["why"])

    if recommendation["inspection_required"]:
        st.markdown(
            '<div class="warning"><b>Professional inspection recommended:</b> '
            "The app does not prescribe a structural repair solely from an image.</div>",
            unsafe_allow_html=True,
        )

    if recommendation["safety_override"]:
        st.markdown(
            '<div class="danger"><b>Safety override:</b> Keep people away from any visibly unsafe area and seek urgent qualified professional assessment. '
            "If there is immediate danger of collapse, follow local emergency procedures.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("## 07 — Final report")
    pdf = make_pdf(result, info, recommendation)
    st.download_button(
        "📄 Download preliminary PDF report",
        data=pdf,
        file_name=f"structure_doctor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    if st.button("💾 Save inspection to history", use_container_width=True):
        save_inspection(result, info)
        st.success("Inspection saved to local SQLite history.")


def track_page():
    st.title("📈 Track Crack Change")
    st.caption("Compare two images taken at different times. Consistent angle, scale and lighting improve comparison.")

    c1, c2 = st.columns(2)
    with c1:
        before_file = st.file_uploader("BEFORE image", type=["jpg", "jpeg", "png"], key="before")
    with c2:
        after_file = st.file_uploader("AFTER image", type=["jpg", "jpeg", "png"], key="after")

    if before_file and after_file:
        before = cv_analyze(before_file)
        after = cv_analyze(after_file)

        b1, b2 = st.columns(2)
        with b1:
            st.markdown("**BEFORE**")
            st.image(before["overlay"], use_container_width=True)
        with b2:
            st.markdown("**AFTER**")
            st.image(after["overlay"], use_container_width=True)

        # Pixel-space comparison is deliberately labeled as such.
        length_change = after["length_px"] - before["length_px"]
        width_change = after["width_px"] - before["width_px"]

        st.markdown("### Visible change estimate")
        k1, k2, k3 = st.columns(3)
        k1.metric("Before length", f'{before["length_px"]:.0f} px')
        k2.metric("After length", f'{after["length_px"]:.0f} px', f'{length_change:+.0f} px')
        k3.metric("After width proxy", f'{after["width_px"]:.1f} px', f'{width_change:+.1f} px')

        if length_change > max(8, before["length_px"] * 0.15) or width_change > max(2, before["width_px"] * 0.20):
            st.warning(
                "The prototype detects a potentially meaningful visible change between the images. "
                "Use consistent scale/positioning and consider professional assessment if the crack is progressing."
            )
        else:
            st.success(
                "No large visible change was detected by this simple comparison. "
                "Continue monitoring; this does not prove structural stability."
            )

        st.info(
            "Tracking is a visual comparison feature, not a validated crack-growth measurement system. "
            "For accurate real-world dimensions, images need a reliable scale/reference and consistent capture conditions."
        )


def history_page():
    st.title("🗂️ Inspection History")
    st.caption("Local SQLite records from this Streamlit prototype.")

    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        """
        SELECT created_at, building_type, element, severity, confidence,
               length_text, width_text, recommendation
        FROM inspections
        ORDER BY id DESC
        """
    ).fetchall()
    con.close()

    if not rows:
        st.info("No saved inspections yet.")
        return

    for row in rows:
        with st.expander(f"{row[0]} — {row[3]} — {row[2]}"):
            st.write(f"**Building:** {row[1]}")
            st.write(f"**Confidence:** {row[4]}%")
            st.write(f"**Estimated length:** {row[5]}")
            st.write(f"**Estimated width:** {row[6]}")
            st.write(f"**Next step:** {row[7]}")


if page == "Home":
    home()
elif page == "Analyze":
    analyze_page()
elif page == "Track Change":
    track_page()
else:
    history_page()

st.markdown("---")
st.caption(
    "Structure Doctor provides preliminary visual assessment only. "
    "It does not replace professional structural inspection, engineering diagnosis, design or certification."
)
