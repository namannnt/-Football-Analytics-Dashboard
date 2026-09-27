"""PDF one-page stakeholder brief."""
from __future__ import annotations

from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def build_pdf(title: str, takeaways: list[str]) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 8 * mm)]
    for takeaway in takeaways:
        story.extend((Paragraph(f"• {takeaway}", styles["BodyText"]), Spacer(1, 4 * mm)))
    document.build(story)
    return buffer.getvalue()
