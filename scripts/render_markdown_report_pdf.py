#!/usr/bin/env python3
"""Render the project's result-report Markdown subset to a Chinese PDF."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    LongTable,
    PageBreak,
    Paragraph,
    KeepTogether,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
    XPreformatted,
)


FONT_PATH = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
FONT_NAME = "DroidSansFallback"
LATIN_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
LATIN_BOLD_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
LATIN_FONT_NAME = "DejaVuSans"
LATIN_BOLD_FONT_NAME = "DejaVuSans-Bold"


def mixed_plain(text: str, *, code: bool = False) -> str:
    chunks = []
    start = 0
    current_latin = None
    for idx, char in enumerate(text + "\0"):
        is_latin = char != "\0" and ord(char) < 0x2E80
        if current_latin is None:
            current_latin = is_latin
            start = idx
            continue
        if char == "\0" or is_latin != current_latin:
            value = html.escape(text[start:idx])
            if current_latin and value:
                color = ' color="#7a1f1f"' if code else ""
                value = f'<font name="{LATIN_FONT_NAME}"{color}>{value}</font>'
            chunks.append(value)
            start = idx
            current_latin = is_latin
    return "".join(chunks)


def inline_markup(text: str) -> str:
    value = text.strip()
    output = []
    position = 0
    token_pattern = re.compile(r"(`[^`]+`|\*\*.+?\*\*)")
    for match in token_pattern.finditer(value):
        output.append(mixed_plain(value[position : match.start()]))
        token = match.group(0)
        if token.startswith("`"):
            output.append(mixed_plain(token[1:-1], code=True))
        else:
            output.append(f"<b>{mixed_plain(token[2:-2])}</b>")
        position = match.end()
    output.append(mixed_plain(value[position:]))
    return "".join(output)


def table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_table_separator(line: str) -> bool:
    cells = table_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def column_widths(rows: list[list[str]], available: float) -> list[float]:
    count = max(len(row) for row in rows)
    weights = []
    for col in range(count):
        lengths = [len(row[col]) if col < len(row) else 0 for row in rows]
        weight = max(5, min(max(lengths, default=5), 26))
        weights.append(weight)
    total = sum(weights)
    return [available * weight / total for weight in weights]


def build_styles():
    base = getSampleStyleSheet()
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))
    pdfmetrics.registerFont(TTFont(LATIN_FONT_NAME, str(LATIN_FONT_PATH)))
    pdfmetrics.registerFont(TTFont(LATIN_BOLD_FONT_NAME, str(LATIN_BOLD_FONT_PATH)))
    pdfmetrics.registerFontFamily(
        FONT_NAME,
        normal=FONT_NAME,
        bold=FONT_NAME,
        italic=FONT_NAME,
        boldItalic=FONT_NAME,
    )
    pdfmetrics.registerFontFamily(
        LATIN_FONT_NAME,
        normal=LATIN_FONT_NAME,
        bold=LATIN_BOLD_FONT_NAME,
        italic=LATIN_FONT_NAME,
        boldItalic=LATIN_BOLD_FONT_NAME,
    )

    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName=FONT_NAME,
            fontSize=20,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#183153"),
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "Heading2CN",
            parent=base["Heading2"],
            fontName=FONT_NAME,
            fontSize=14,
            leading=20,
            textColor=colors.HexColor("#183153"),
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "Heading3CN",
            parent=base["Heading3"],
            fontName=FONT_NAME,
            fontSize=11.5,
            leading=17,
            textColor=colors.HexColor("#315b7d"),
            spaceBefore=9,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=9.3,
            leading=15,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#222222"),
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "BulletCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=9.2,
            leading=14.5,
            leftIndent=13,
            firstLineIndent=-8,
            bulletIndent=3,
            spaceAfter=3,
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "CodeCN",
            parent=base["Code"],
            fontName=FONT_NAME,
            fontSize=7.8,
            leading=11.5,
            leftIndent=6,
            rightIndent=6,
            borderColor=colors.HexColor("#d7dde5"),
            borderWidth=0.5,
            borderPadding=6,
            backColor=colors.HexColor("#f6f8fa"),
            spaceBefore=3,
            spaceAfter=7,
            wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "TableHeaderCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=7.6,
            leading=10,
            textColor=colors.white,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "table_cell": ParagraphStyle(
            "TableCellCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=7.3,
            leading=10,
            textColor=colors.HexColor("#222222"),
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
    }


def render_markdown(source: Path, destination: Path) -> None:
    if not FONT_PATH.is_file():
        raise FileNotFoundError(f"Chinese font not found: {FONT_PATH}")

    styles = build_styles()
    page_width, _ = A4
    left_margin = right_margin = 17 * mm
    available = page_width - left_margin - right_margin
    doc = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=18 * mm,
        bottomMargin=17 * mm,
        title="ST-WAM 三任务 LIBERO-Plus 消融实验报告",
        author="ST-WAM reproduction experiment",
        subject="LIBERO-Plus ablation results",
    )

    lines = source.read_text(encoding="utf-8").splitlines()
    story = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            story.append(Paragraph(inline_markup(" ".join(paragraph)), styles["body"]))
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            story.append(XPreformatted(inline_markup("\n".join(code_lines)), styles["code"]))
        elif stripped.startswith("|") and i + 1 < len(lines) and is_table_separator(lines[i + 1]):
            flush_paragraph()
            raw_rows = [table_cells(line)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                raw_rows.append(table_cells(lines[i]))
                i += 1
            i -= 1
            cols = max(len(row) for row in raw_rows)
            raw_rows = [row + [""] * (cols - len(row)) for row in raw_rows]
            data = []
            for row_idx, row in enumerate(raw_rows):
                style = styles["table_header"] if row_idx == 0 else styles["table_cell"]
                data.append([Paragraph(inline_markup(cell), style) for cell in row])
            table = LongTable(data, colWidths=column_widths(raw_rows, available), repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#315b7d")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b9c3cf")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                    + [
                        ("BACKGROUND", (0, row), (-1, row), colors.HexColor("#f3f6f9"))
                        for row in range(2, len(data), 2)
                    ]
                )
            )
            table_block = [table, Spacer(1, 6)]
            if len(data) <= 8:
                story.append(KeepTogether(table_block))
            else:
                story.extend(table_block)
        elif stripped.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[2:]), styles["title"]))
            story.append(Spacer(1, 4))
        elif stripped.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[3:]), styles["h2"]))
        elif stripped.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[4:]), styles["h3"]))
        elif re.match(r"^[-*]\s+", stripped):
            flush_paragraph()
            story.append(Paragraph(inline_markup(re.sub(r"^[-*]\s+", "", stripped)), styles["bullet"], bulletText="•"))
        elif re.match(r"^\d+\.\s+", stripped):
            flush_paragraph()
            marker, content = stripped.split(".", 1)
            story.append(Paragraph(inline_markup(content.strip()), styles["bullet"], bulletText=f"{marker}."))
        elif stripped == "---":
            flush_paragraph()
            story.append(Spacer(1, 4))
        elif not stripped:
            flush_paragraph()
        else:
            paragraph.append(stripped.rstrip("  "))
        i += 1

    flush_paragraph()

    def decorate_page(canvas, document):
        canvas.saveState()
        canvas.setFont(FONT_NAME, 7.5)
        canvas.setFillColor(colors.HexColor("#667085"))
        if document.page > 1:
            canvas.drawString(left_margin, A4[1] - 10 * mm, "三任务消融实验报告")
        canvas.setFont(LATIN_FONT_NAME, 7.5)
        canvas.drawRightString(A4[0] - right_margin, 9 * mm, str(document.page))
        canvas.restoreState()

    destination.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=decorate_page, onLaterPages=decorate_page)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    render_markdown(args.source.resolve(), args.destination.resolve())
    print(args.destination.resolve())


if __name__ == "__main__":
    main()
