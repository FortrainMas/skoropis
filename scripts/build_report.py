"""Bundle docs/plan.md, docs/validation.md and the experiment records into docs/report.pdf.

    uv run python scripts/build_report.py

Markdown stays the source of truth; the PDF is a snapshot for reading and
sending. Needs pandoc and Chrome (or Edge); set CHROME to override the path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Segoe UI", Inter, Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.45;
       color: #0b0b0b; max-width: none; margin: 0; padding: 0; }
h1 { font-size: 20pt; margin: 0 0 6pt; break-before: page; }
header + h1, h1:first-of-type { break-before: avoid; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
code { font-family: "Cascadia Mono", Consolas, monospace; font-size: 9pt; background: #f0efec;
       padding: 0 2pt; border-radius: 2pt; }
pre { background: #f0efec; padding: 6pt 8pt; border-radius: 4pt; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9.5pt; break-inside: avoid; }
th, td { border-bottom: 1px solid #d4d3cf; padding: 3pt 6pt; vertical-align: top; }
td { font-variant-numeric: tabular-nums; }
th { border-bottom: 1.5px solid #8a8985; text-align: left; }
img { max-width: 100%; display: block; margin: 8pt auto; break-inside: avoid; }
figure { margin: 0; break-inside: avoid; }
figcaption { font-size: 9pt; color: #52514e; text-align: center; }
header { margin-bottom: 14pt; }
header .title { font-size: 24pt; margin: 0; }
header .date { color: #52514e; }
a { color: #1c5cab; text-decoration: none; }
"""


def find_chrome() -> str:
    candidates = [
        os.environ.get("CHROME"),
        shutil.which("chrome"),
        shutil.which("msedge"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("Chrome or Edge not found; set CHROME to its path")


def main() -> None:
    sources = [DOCS / "plan.md", DOCS / "validation.md", DOCS / "experiments" / "README.md"]
    sources += sorted((DOCS / "experiments").glob("E*.md"))
    out = DOCS / "report.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        css, html = Path(tmp) / "report.css", Path(tmp) / "report.html"
        css.write_text(CSS, encoding="utf-8")
        subprocess.run(
            [
                "pandoc", *map(str, sources),
                "--from", "gfm+tex_math_dollars", "--to", "html5", "--standalone", "--embed-resources",
                "--mathml", "--css", str(css),
                "--resource-path", os.pathsep.join([str(DOCS), str(DOCS / "experiments")]),
                "--metadata", "title=Распознавание рукописей: валидация и эксперименты",
                "--metadata", f"date={date.today().isoformat()}",
                "--metadata", "lang=ru",
                "--output", str(html),
            ],
            check=True,
        )  # fmt: skip
        subprocess.run(
            [
                find_chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={out}", html.as_uri(),
            ],
            check=True, capture_output=True,
        )  # fmt: skip
    print(out)


if __name__ == "__main__":
    main()
