import re
from fpdf import FPDF


def sanitize_text(text: str) -> str:
    """
    Makes text safe for FPDF's built-in Helvetica font.
    Removes markdown, emojis, problematic Unicode characters,
    and normalizes whitespace.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # Remove markdown styling
    text = text.replace("**", "").replace("*", "")

    # Replace common symbols/emojis with safe text
    replacements = {
        "⚠️": "[WARNING]",
        "⚠": "[WARNING]",
        "🚨": "[ALERT]",
        "🔧": "[MAINTENANCE]",
        "🛑": "[CRITICAL]",
        "→": "->",
        "←": "<-",
        "≥": ">=",
        "≤": "<=",
        "–": "-",
        "—": "-",
        "•": "-",
        "°": " deg ",
        "μ": "u",
        "×": "x",
        "±": "+/-",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove remaining characters unsupported by Helvetica
    text = text.encode("latin-1", "ignore").decode("latin-1")

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def safe_multicell(pdf, text, height=6):
    """
    Safely render multi-line text using the available page width.

    Explicitly calculates the available width instead of relying on
    multi_cell(width=0), which can cause:
    'Not enough horizontal space to render a single character'
    """
    text = sanitize_text(text)

    if not text:
        return

    # Available width between left and right margins
    available_width = (
        pdf.w
        - pdf.l_margin
        - pdf.r_margin
    )

    # Safety margin to prevent floating-point/cursor edge cases
    available_width = max(10, available_width - 2)

    # Make sure the cursor starts at the left margin
    pdf.set_x(pdf.l_margin)

    pdf.multi_cell(
        w=available_width,
        h=height,
        text=text,
        new_x="LMARGIN",
        new_y="NEXT",
    )


class PDFReport(FPDF):

    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_x(self.l_margin)

        self.cell(
            w=self.w - self.l_margin - self.r_margin,
            h=10,
            text="IBM Predictive Maintenance Diagnostic Report",
            border=False,
            new_x="LMARGIN",
            new_y="NEXT",
            align="C",
        )

        self.set_font("Helvetica", "", 9)
        self.set_x(self.l_margin)

        self.cell(
            w=self.w - self.l_margin - self.r_margin,
            h=5,
            text="Automated Equipment Telemetry & Risk Inspection Assessment",
            border=False,
            new_x="LMARGIN",
            new_y="NEXT",
            align="C",
        )

        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)

        self.cell(
            w=0,
            h=10,
            text=f"Page {self.page_no()}",
            align="C",
        )


def generate_pdf_report(
    prod_type,
    air_temp,
    proc_temp,
    rot_speed,
    torque,
    tool_wear,
    failure_prob,
    reasons,
):
    pdf = PDFReport()

    # A4 page with 15 mm margins
    pdf.set_margins(15, 15, 15)

    pdf.set_auto_page_break(
        auto=True,
        margin=15,
    )

    pdf.add_page()

    # ---------------------------------------------------------
    # SAFE AVAILABLE PAGE WIDTH
    # ---------------------------------------------------------
    page_width = (
        pdf.w
        - pdf.l_margin
        - pdf.r_margin
    )

    # ---------------------------------------------------------
    # HANDLE FAILURE PROBABILITY
    # ---------------------------------------------------------
    try:
        failure_prob = float(failure_prob)
    except (TypeError, ValueError):
        failure_prob = 0.0

    prob_val = (
        failure_prob * 100
        if failure_prob <= 1.0
        else failure_prob
    )

    # Keep probability within a sensible display range
    prob_val = max(0.0, min(100.0, prob_val))

    # ---------------------------------------------------------
    # 1. STATUS BANNER
    # ---------------------------------------------------------
    pdf.set_font("Helvetica", "B", 12)

    if prob_val >= 50.0:
        pdf.set_fill_color(255, 235, 235)
        pdf.set_text_color(200, 0, 0)

        status_text = (
            f"CRITICAL ALERT: HIGH FAILURE RISK "
            f"({prob_val:.1f}%)"
        )
    else:
        pdf.set_fill_color(235, 255, 235)
        pdf.set_text_color(0, 128, 0)

        status_text = (
            f"OPERATIONAL NORMAL: LOW FAILURE RISK "
            f"({prob_val:.1f}%)"
        )

    pdf.set_x(pdf.l_margin)

    pdf.cell(
        w=page_width,
        h=10,
        text=sanitize_text(status_text),
        border=1,
        fill=True,
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    pdf.ln(5)

    # Reset text color
    pdf.set_text_color(0, 0, 0)

    # ---------------------------------------------------------
    # 2. TELEMETRY SUMMARY TABLE
    # ---------------------------------------------------------
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_x(pdf.l_margin)

    pdf.cell(
        w=page_width,
        h=8,
        text="1. Sensor Telemetry Snapshot",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    metrics = [
        (
            "Product Quality Type",
            sanitize_text(prod_type),
        ),
        (
            "Air Temperature [K]",
            f"{float(air_temp):.1f} K",
        ),
        (
            "Process Temperature [K]",
            f"{float(proc_temp):.1f} K",
        ),
        (
            "Rotational Speed [rpm]",
            f"{rot_speed} RPM",
        ),
        (
            "Torque [Nm]",
            f"{float(torque):.1f} Nm",
        ),
        (
            "Tool Wear [min]",
            f"{tool_wear} min",
        ),
    ]

    pdf.set_font("Helvetica", "", 10)

    # Use calculated width rather than hard-coded 180.
    # A4 width = 210 mm, margins = 15 + 15, so available = 180 mm.
    col_width = page_width / 2

    with pdf.table(
        width=page_width,
        col_widths=(col_width, col_width),
        align="CENTER",
    ) as table:

        for label, val in metrics:
            row = table.row()

            row.cell(sanitize_text(label))
            row.cell(sanitize_text(val))

    pdf.ln(5)

    # ---------------------------------------------------------
    # 3. DIAGNOSTIC THRESHOLD FINDINGS
    # ---------------------------------------------------------
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_x(pdf.l_margin)

    pdf.cell(
        w=page_width,
        h=8,
        text="2. Diagnostic Threshold Findings",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    pdf.set_font("Helvetica", "", 10)

    if reasons:

        # Make sure reasons is iterable
        if isinstance(reasons, str):
            reasons = [reasons]

        for r in reasons:

            clean_r = sanitize_text(r)

            if not clean_r:
                continue

            # Prefix using ASCII hyphen rather than Unicode bullet
            reason_text = f"- {clean_r}"

            safe_multicell(
                pdf,
                reason_text,
                height=6,
            )

    else:

        safe_multicell(
            pdf,
            "No critical physical thresholds exceeded. "
            "All sensors operating within normal parameters.",
            height=6,
        )

    # ---------------------------------------------------------
    # 4. REPORT FOOTER INFORMATION
    # ---------------------------------------------------------
    pdf.ln(5)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)

    safe_multicell(
        pdf,
        "Generated automatically by the IBM Predictive "
        "Maintenance Diagnostic System.",
        height=5,
    )

    # Return PDF as bytes
    return bytes(pdf.output())