"""Render docs/manuscript.md to a PDF with the exhibits in place.

Run from the project folder:  python scripts/render_pdf.py

Needs the Python packages `markdown` and `playwright` (with its Chromium). This is the
The output is written to the project root as AI_Data_Centres_Investment_Thesis.pdf.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SRC = DOCS / "manuscript.md"
HTML = DOCS / "manuscript.html"
PDF = ROOT / "AI_Data_Centres_Investment_Thesis.pdf"

CSS = """
@font-face { font-family: Plex Sans; src: url('../assets/fonts/IBMPlexSans-Regular.ttf'); font-weight: 400; }
@font-face { font-family: Plex Sans; src: url('../assets/fonts/IBMPlexSans-Italic.ttf'); font-weight: 400; font-style: italic; }
@font-face { font-family: Plex Sans; src: url('../assets/fonts/IBMPlexSans-SemiBold.ttf'); font-weight: 600; }
@font-face { font-family: Plex Serif; src: url('../assets/fonts/IBMPlexSerif-SemiBold.ttf'); font-weight: 600; }
@font-face { font-family: Plex Serif; src: url('../assets/fonts/IBMPlexSerif-Regular.ttf'); font-weight: 400; }
@page { size: A4; margin: 22mm 22mm 20mm 22mm; }
:root { --green: #0e7c55; --green-dark: #0b4f39; --ink: #15201b; --ink2: #4b5651; --rule: #dfe5e2; }
body { font-family: 'Plex Sans', sans-serif; font-size: 9.8pt; line-height: 1.55; color: var(--ink); margin: 0; }
.cover { height: 230mm; display: flex; flex-direction: column; justify-content: center; page-break-after: always; }
.cover .kicker { font-size: 9pt; letter-spacing: .12em; text-transform: uppercase; color: var(--green-dark); font-weight: 600; }
.cover .title { font-family: 'Plex Serif', serif; font-weight: 600; font-size: 30pt; line-height: 1.12; margin: 10mm 0 8mm; color: var(--ink); }
.cover .meta { font-size: 10.5pt; color: var(--ink2); border-top: 1.2pt solid var(--green-dark); padding-top: 4mm; }
h1 { font-family: 'Plex Serif', serif; font-weight: 600; font-size: 17pt; color: var(--ink); margin: 0 0 4mm;
     padding-bottom: 2mm; border-bottom: 1pt solid var(--green-dark); page-break-before: always; page-break-after: avoid; }
h2 { font-family: 'Plex Serif', serif; font-weight: 600; font-size: 12.5pt; margin: 6mm 0 2mm; page-break-after: avoid; }
p { margin: 0 0 2.6mm; text-align: left; orphans: 3; widows: 3; }
strong { font-weight: 600; color: var(--ink); }
ul { margin: 0 0 3mm; padding-left: 5mm; } li { margin-bottom: 1.4mm; }
img { width: 100%; display: block; margin: 4mm 0 5mm; page-break-inside: avoid; break-inside: avoid; }
p:has(> img) { page-break-inside: avoid; break-inside: avoid; margin: 0; }
table { border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: 2mm 0 4mm; page-break-inside: avoid; }
th { text-align: left; font-weight: 600; color: var(--ink2); border-bottom: 1pt solid var(--green-dark); padding: 1.6mm 2mm; }
td { border-bottom: .5pt solid var(--rule); padding: 1.4mm 2mm; vertical-align: top; }
.keybox { background: #f2f6f4; border-top: 1.2pt solid var(--green-dark); border-bottom: .6pt solid var(--green-dark); padding: 2mm 4mm .5mm; margin: 2mm 0 4mm; page-break-inside: avoid; }
.keybox-title { font-size: 8pt; letter-spacing: .1em; text-transform: uppercase; font-weight: 600; color: var(--green-dark); margin-bottom: 1mm; }
.keybox table { margin: 0; font-size: 8.8pt; }
.keybox td { border-bottom: .5pt solid #d6e0db; padding: 1mm 0; line-height: 1.4; }
.keybox tr:last-child td { border-bottom: none; }
.keybox td:first-child { color: var(--ink2); width: 48%; padding-right: 4mm; }
.keybox td:last-child { font-weight: 600; }
.disclaimer { margin-top: 12mm; padding-top: 3mm; border-top: .6pt solid var(--rule); font-size: 7.8pt; line-height: 1.45; color: var(--ink2); page-break-inside: avoid; }
.disclaimer-title { font-weight: 600; text-transform: uppercase; letter-spacing: .1em; font-size: 7.5pt; margin-bottom: 1mm; }
code { font-family: 'Plex Sans', sans-serif; font-size: 9pt; background: #f1f4f2; padding: 0 1mm; border-radius: 1mm; }
"""

FOOTER = ("<div style=\"font-family: sans-serif; font-size: 7pt; color: #7d8784; width: 100%; padding: 0 22mm; "
          "display: flex; justify-content: space-between;\"><span>{title}</span>"
          "<span><span class=\"pageNumber\"></span></span></div>")


def front_matter(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}, text
    meta = dict(re.findall(r'^(\w+):\s*"(.*)"$', m.group(1), re.M))
    return meta, text[m.end():]


def build_html() -> dict:
    meta, body = front_matter(SRC.read_text(encoding="utf-8"))
    html = markdown.markdown(body, extensions=["tables"])
    cover = (f'<section class="cover"><div class="kicker">Investment thesis</div>'
             f'<div class="title">{meta.get("title", "")}</div>'
             f'<div class="meta">{meta.get("author", "")}<br>{meta.get("date", "")}</div></section>')
    HTML.write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>{meta.get("title", "")}</title>'
                    f'<style>{CSS}</style></head><body>{cover}{html}</body></html>', encoding="utf-8")
    return meta


def main() -> None:
    meta = build_html()
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        kw = {}
        if Path("/opt/pw-browsers/chromium").exists():
            kw["executable_path"] = "/opt/pw-browsers/chromium"
        browser = pw.chromium.launch(**kw)
        page = browser.new_page()
        page.goto(HTML.as_uri())
        page.wait_for_load_state("networkidle")
        page.pdf(path=str(PDF), format="A4", print_background=True, display_header_footer=True,
                 header_template="<div></div>", footer_template=FOOTER.format(title=meta.get("title", "")),
                 margin={"top": "22mm", "bottom": "20mm", "left": "22mm", "right": "22mm"})
        browser.close()
    print(f"Wrote {PDF.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
