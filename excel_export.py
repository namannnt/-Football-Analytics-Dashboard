"""Multi-sheet Excel export for curated analytics."""
from __future__ import annotations

from io import BytesIO


def build_excel(sheets: dict[str, object]) -> bytes:
    buffer = BytesIO()
    with __import__("pandas").ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
            sheet = writer.book[name[:31]]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                width = min(max(len(str(cell.value or "")) for cell in column) + 2, 40)
                sheet.column_dimensions[column[0].column_letter].width = width
    return buffer.getvalue()
