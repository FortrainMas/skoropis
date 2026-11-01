"""SVG figures for docs/, generated from data and evaluation runs (never drawn by hand).

    uv run python scripts/report_figures.py splits
    uv run python scripts/report_figures.py run kansallisarkisto-zeroshot --prefix E001

Static figures for Markdown and PDF, so no interactivity; colors are the first
categorical slots of a CVD-validated palette, and every series is also labeled
directly, so identity never relies on color alone.
"""

from __future__ import annotations

import argparse
import base64
import json
from html import escape
from io import BytesIO
from pathlib import Path

from PIL import Image

from diploma.data import REPO_ROOT, SPLITS, Sample, load_dataset
from diploma.metrics import Tally, align
from diploma.text import Norm, normalize

FIGURES = REPO_ROOT / "docs" / "figures"
RUNS = REPO_ROOT / "runs"

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e8e7e3"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
FONT = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"
SERIF = "'Times New Roman', 'Noto Serif', serif"  # has proper lowercase ѣ, unlike most UI fonts

NORM_LABELS = {
    Norm.EXACT: ("exact", "как есть"),
    Norm.CASELESS: ("caseless", "без регистра"),
    Norm.LETTERS: ("letters", "без пунктуации"),
    Norm.MODERN: ("modern", "совр. орфография"),
}


def svg(width: int, height: int, title: str, body: list[str]) -> str:
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'font-family="{FONT}">',
        f"  <title>{escape(title)}</title>",
        f'  <rect width="{width}" height="{height}" fill="{SURFACE}"/>',
        *(f"  {line}" for line in body),
        "</svg>\n",
    ])  # fmt: skip


def text(x: float, y: float, content: str, *, size: float = 12, fill: str = INK_2, anchor: str = "start",
         weight: int = 400, family: str | None = None) -> str:  # fmt: skip
    font = f' font-family="{family}"' if family else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}"{font}>{escape(content)}</text>'
    )


def bar(x: float, y: float, width: float, height: float, fill: str, radius: float = 4) -> str:
    """A vertical bar with rounded data end (top) and square baseline."""
    r = min(radius, width / 2, height)
    return (
        f'<path d="M{x:.1f},{y + height:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} '
        f'H{x + width - r:.1f} Q{x + width:.1f},{y:.1f} {x + width:.1f},{y + r:.1f} V{y + height:.1f} Z" '
        f'fill="{fill}"/>'
    )


def legend(x: float, y: float, entries: list[tuple[str, str]]) -> list[str]:
    out = []
    for label, color in entries:
        out.append(f'<rect x="{x:.1f}" y="{y - 9:.1f}" width="10" height="10" rx="2" fill="{color}"/>')
        out.append(text(x + 16, y, label, fill=INK))
        x += 26 + 6.6 * len(label)
    return out


# --- data splits -----------------------------------------------------------


def splits_figure() -> str:
    rows = []
    for name, title, note in [
        ("makarov", "Макаровка — сплит по страницам", "каждая страница целиком в одной части"),
        ("peter", "Digital Peter — официальный сплит соревнования", "сплит построчный: страницы встречаются в разных частях"),
    ]:  # fmt: skip
        samples = load_dataset(name)
        counts = {s: sum(1 for x in samples if x.split == s) for s in SPLITS}
        pages = {s: len({x.group for x in samples if x.split == s}) for s in SPLITS}
        rows.append((title, note, counts, pages))

    width, left, bar_width, bar_height = 760, 20, 720, 26
    body: list[str] = []
    y = 34
    for title, note, counts, pages in rows:
        body.append(text(left, y, title, size=14, fill=INK, weight=600))
        total = sum(counts.values())
        clip = f"clip-{y}"
        body.append(f'<clipPath id="{clip}"><rect x="{left}" y="{y + 12}" width="{bar_width}" height="{bar_height}" rx="4"/></clipPath>')
        x = left
        for i, split in enumerate(SPLITS):
            share = counts[split] / total * bar_width
            gap = 2 if i < len(SPLITS) - 1 else 0
            body.append(
                f'<rect x="{x:.1f}" y="{y + 12}" width="{max(share - gap, 0):.1f}" height="{bar_height}" '
                f'fill="{SERIES[i]}" clip-path="url(#{clip})"/>'
            )
            middle = x + share / 2
            body.append(text(middle, y + 12 + bar_height + 18, split, fill=INK, anchor="middle", weight=600))
            body.append(text(middle, y + 12 + bar_height + 34, f"{counts[split]} строк · {pages[split]} стр.",
                             size=11.5, anchor="middle"))  # fmt: skip
            x += share
        body.append(text(left, y + 12 + bar_height + 56, note, size=11.5, fill=MUTED))
        y += 120
    return svg(width, y - 10, "Разбиение датасетов на train / val / test", body)


# --- one evaluation run ----------------------------------------------------


def load_run(run: str) -> list[tuple[str, dict, list[dict]]]:
    out = []
    for path in sorted((RUNS / run).glob("*/metrics.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        lines = [json.loads(l) for l in (path.parent / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        out.append((dataset_label(report), report, lines))
    return out


def dataset_label(report: dict) -> str:
    name, _, split = report["dataset"].partition(":")
    label = {"makarov": "Макаровка", "peter": "Digital Peter"}.get(name, name)
    label += f", {split}" if split else ""
    if report.get("sample"):
        label += f" (выборка {report['sample']})"
    return label


def cer_by_norm_figure(runs: list[tuple[str, dict, list[dict]]]) -> str:
    width, height = 760, 380
    left, right, top, bottom = 56, 16, 62, 64
    plot_w, plot_h = width - left - right, height - top - bottom
    highest = max(r["metrics"][n.value]["ci95"]["cer"][1] for _, r, _ in runs for n in Norm)
    y_max = max(10, -(-highest * 100 // 10) * 10)
    scale = lambda v: top + plot_h * (1 - v * 100 / y_max)  # noqa: E731

    body = [text(left, 22, "CER при разных нормализациях, 95% доверительный интервал", size=14, fill=INK, weight=600)]
    body += legend(left, 46, [(label, SERIES[i]) for i, (label, _, _) in enumerate(runs)])
    for tick in range(0, int(y_max) + 1, 10):
        y = scale(tick / 100)
        body.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}"/>')
        body.append(text(left - 8, y + 4, f"{tick}%", size=11, fill=MUTED, anchor="end"))

    group_w, bar_w = plot_w / len(Norm), 44
    for g, norm in enumerate(Norm):
        center = left + group_w * (g + 0.5)
        first = center - (bar_w * len(runs) + 2 * (len(runs) - 1)) / 2
        for i, (_, report, _) in enumerate(runs):
            scores = report["metrics"][norm.value]
            x, y = first + i * (bar_w + 2), scale(scores["cer"])
            body.append(bar(x, y, bar_w, scale(0) - y, SERIES[i]))
            low, high = (scale(v) for v in scores["ci95"]["cer"])
            mid = x + bar_w / 2
            body.append(f'<path d="M{mid - 5:.1f},{high:.1f} H{mid + 5:.1f} M{mid:.1f},{high:.1f} V{low:.1f} '
                        f'M{mid - 5:.1f},{low:.1f} H{mid + 5:.1f}" stroke="{INK}" stroke-width="1.25" fill="none"/>')  # fmt: skip
            body.append(text(mid, high - 6, f"{scores['cer'] * 100:.1f}", size=11.5, fill=INK, anchor="middle"))
        name, hint = NORM_LABELS[norm]
        body.append(text(center, height - bottom + 22, name, fill=INK, anchor="middle", weight=600))
        body.append(text(center, height - bottom + 38, hint, size=11.5, anchor="middle"))
    return svg(width, height, "CER по уровням нормализации", body)


def coverage_figure(runs: list[tuple[str, dict, list[dict]]], mark: float = 0.2) -> str:
    width, height = 760, 380
    left, right, top, bottom = 56, 150, 62, 52
    plot_w, plot_h = width - left - right, height - top - bottom

    curves = []
    for label, _, lines in runs:
        ranked = sorted(lines, key=lambda l: -l["confidence"])
        errors = chars = 0
        points = []
        for k, line in enumerate(ranked, 1):
            tally = Tally.of_line(line["ref"], line["hyp"])
            errors, chars = errors + tally.chars.errors, chars + tally.chars.ref_len
            points.append((k / len(ranked), errors / chars))
        curves.append((label, points))
    highest = max(c for _, pts in curves for f, c in pts if f >= 0.05)
    y_max = max(10, -(-highest * 100 // 10) * 10)
    sx = lambda f: left + plot_w * f  # noqa: E731
    sy = lambda v: top + plot_h * (1 - v * 100 / y_max)  # noqa: E731

    body = [text(left, 22, "CER на строках, где модель увереннее всего", size=14, fill=INK, weight=600)]
    body += legend(left, 46, [(label, SERIES[i]) for i, (label, _) in enumerate(curves)])
    for tick in range(0, int(y_max) + 1, 10):
        body.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{sy(tick / 100):.1f}" y2="{sy(tick / 100):.1f}" stroke="{GRID}"/>')
        body.append(text(left - 8, sy(tick / 100) + 4, f"{tick}%", size=11, fill=MUTED, anchor="end"))
    for tick in range(0, 101, 20):
        body.append(text(sx(tick / 100), top + plot_h + 18, f"{tick}%", size=11, fill=MUTED, anchor="middle"))
    body.append(text(left + plot_w / 2, height - 10, "доля строк, отобранных по убыванию уверенности", size=12,
                     anchor="middle"))  # fmt: skip
    body.append(f'<line x1="{sx(mark):.1f}" x2="{sx(mark):.1f}" y1="{top}" y2="{top + plot_h}" stroke="{MUTED}" '
                f'stroke-dasharray="4 4"/>')  # fmt: skip
    body.append(text(sx(mark) + 6, top + 12, "20% — столько статья добавляет за итерацию", size=11, fill=MUTED))

    for i, (label, points) in enumerate(curves):
        shown = [(f, c) for f, c in points if f >= 0.05]
        path = " ".join(f"{'M' if j == 0 else 'L'}{sx(f):.1f},{sy(c):.1f}" for j, (f, c) in enumerate(shown))
        body.append(f'<path d="{path}" fill="none" stroke="{SERIES[i]}" stroke-width="2" stroke-linejoin="round"/>')
        at_mark = min(points, key=lambda p: abs(p[0] - mark))
        body.append(f'<circle cx="{sx(at_mark[0]):.1f}" cy="{sy(at_mark[1]):.1f}" r="4.5" fill="{SERIES[i]}" '
                    f'stroke="{SURFACE}" stroke-width="2"/>')  # fmt: skip
        body.append(text(sx(at_mark[0]) - 8, sy(at_mark[1]) - 8, f"{at_mark[1] * 100:.1f}%", size=11.5, fill=INK,
                         anchor="end"))  # fmt: skip
        end = shown[-1]
        body.append(text(sx(end[0]) + 8, sy(end[1]) + 4, f"{end[1] * 100:.1f}% — все строки", size=11.5, fill=INK))
    return svg(width, height, "Кривая риск—покрытие по уверенности модели", body)


ERROR_TINT, SPELLING_TINT = "#d03b3b", "#fab219"
SPELLING_ONLY = {("ѣ", "е"), ("i", "и")}


def predictions(run: str, dataset: str) -> list[dict]:
    path = RUNS / run / dataset.replace(":", "_") / "predictions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def line_picture(sample: Sample, x: float, y: float, width: int) -> tuple[str, int]:
    """The line image as an embedded JPEG ``width`` px wide; returns the element and its height."""
    with Image.open(sample.image) as image:
        image = image.convert("RGB")
        height = round(image.height * width / image.width)
        buffer = BytesIO()
        image.resize((width * 2, height * 2), Image.LANCZOS).save(buffer, "JPEG", quality=80)
    data = base64.b64encode(buffer.getvalue()).decode()
    return f'<image x="{x}" y="{y}" width="{width}" height="{height}" href="data:image/jpeg;base64,{data}"/>', height


def aligned_rows(ref: str, hyp: str, x: float, y: float, label_w: float = 76) -> tuple[list[str], float]:
    """Reference over hypothesis, character by character, errors tinted; returns elements and width."""
    edits = align(ref, hyp)
    # Narrow columns for matched spaces, so words still read as words.
    widths = [10 if e.op == "=" and e.ref == " " else 19 for e in edits]
    body = [text(x, y, "эталон", size=13), text(x, y + 34, "модель", size=13)]
    cx = x + label_w
    for e, cell in zip(edits, widths):
        if e.op != "=":
            tint = SPELLING_TINT if (e.ref, e.hyp) in SPELLING_ONLY else ERROR_TINT
            body.append(f'<rect x="{cx + 1:.0f}" y="{y - 21:.0f}" width="{cell - 2}" height="58" rx="4" '
                        f'fill="{tint}" fill-opacity="0.22"/>')  # fmt: skip
        for row, char in enumerate((e.ref, e.hyp)):
            shown = "␣" if char == " " and e.op != "=" else (char or "")
            body.append(text(cx + cell / 2, y + row * 34, shown, size=21, fill=INK, anchor="middle", family=SERIF))
        cx += cell
    return body, label_w + sum(widths)


def tint_legend(x: float, y: float) -> list[str]:
    return [
        f'<rect x="{x}" y="{y - 10}" width="12" height="12" rx="2" fill="{ERROR_TINT}" fill-opacity="0.22"/>',
        text(x + 18, y, "ошибка распознавания", size=12.5),
        f'<rect x="{x + 190}" y="{y - 10}" width="12" height="12" rx="2" fill="{SPELLING_TINT}" fill-opacity="0.22"/>',
        text(x + 208, y, "ошибка только в орфографии: в режиме modern не считается", size=12.5),
    ]


def example_figure(run: str, dataset: str, sample_id: str) -> str:
    """One real test line: the image, the model's reading, the aligned errors and the CER they make."""
    line = next(p for p in predictions(run, dataset) if p["id"] == sample_id)
    sample = next(s for s in load_dataset(dataset) if s.id == sample_id)
    left = 24
    rows, rows_w = aligned_rows(line["ref"], line["hyp"], left, 0)
    width = round(left + rows_w + left)
    picture, picture_h = line_picture(sample, left, 16, width - 2 * left)
    y = 16 + picture_h + 40
    body = [picture, *aligned_rows(line["ref"], line["hyp"], left, y)[0]]

    y += 34 + 44
    body += tint_legend(left, y)
    exact = Tally.of_line(line["ref"], line["hyp"]).chars
    modern = Tally.of_line(normalize(line["ref"], Norm.MODERN), normalize(line["hyp"], Norm.MODERN)).chars
    parts = [(exact.substitutions, "замены"), (exact.deletions, "удаления"), (exact.insertions, "вставки")]
    numerator = " + ".join(f"{n} {name}" for n, name in parts if n)
    y += 34
    body.append(text(left, y, f"CER = ({numerator}) / {exact.ref_len} символов эталона = {exact.rate * 100:.1f}%",
                     size=15, fill=INK, weight=600))  # fmt: skip
    body.append(text(left, y + 24, f"в режиме modern: {modern.errors} / {modern.ref_len} = {modern.rate * 100:.1f}%",
                     size=13))  # fmt: skip
    return svg(width, y + 40, f"Как считается CER на строке {sample_id}", body)


def plural(n: int, one: str, few: str, many: str) -> str:
    """Russian noun form for ``n``: 1 ошибка, 3 ошибки, 5 ошибок."""
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} {one}"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} {few}"
    return f"{n} {many}"


def gallery_figure(run: str, dataset: str, levels: list[float], max_chars: int = 38) -> str:
    """For each CER level, the real line whose own CER is closest to it.

    Only lines of 15..``max_chars`` characters are eligible, so a level is not
    illustrated by a two-letter fragment and every row fits the page.
    """
    lines = [p for p in predictions(run, dataset) if 15 <= len(p["ref"]) <= max_chars]
    samples = {s.id: s for s in load_dataset(dataset)}
    picked: list[tuple[float, dict]] = []
    for level in levels:
        free = [p for p in lines if all(p["id"] != q["id"] for _, q in picked)]
        picked.append((level, min(free, key=lambda p: (abs(p["cer"] - level), p["id"]))))

    left, picture_w = 24, 700
    rows = [aligned_rows(p["ref"], p["hyp"], left, 0)[1] for _, p in picked]
    width = round(max(picture_w, *rows) + 2 * left)
    body: list[str] = []
    y = 28
    for level, p in picked:
        exact = Tally.of_line(p["ref"], p["hyp"]).chars
        body.append(text(left, y, f"CER строки {exact.rate * 100:.0f}%", size=15, fill=INK, weight=600))
        errors = plural(exact.errors, "ошибка", "ошибки", "ошибок")
        chars = plural(exact.ref_len, "символ", "символа", "символов")
        body.append(text(left + 150, y, f"{errors} на {chars} · {p['id']} · уверенность модели {p['confidence']:.2f}",
                         size=12.5))  # fmt: skip
        picture, picture_h = line_picture(samples[p["id"]], left, y + 12, picture_w)
        body.append(picture)
        y += 12 + picture_h + 36
        body += aligned_rows(p["ref"], p["hyp"], left, y)[0]
        y += 34 + 56
    body += tint_legend(left, y - 20)
    return svg(width, y + 4, f"Как выглядят строки с разным CER: {dataset}", body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("splits")
    p = commands.add_parser("example")
    p.add_argument("run")
    p.add_argument("dataset")
    p.add_argument("sample_id")
    p = commands.add_parser("gallery")
    p.add_argument("run")
    p.add_argument("dataset")
    p.add_argument("--levels", default="0.05,0.1,0.2,0.4", help="comma-separated CER levels")
    p.add_argument("--out", required=True, help="file name in docs/figures/")
    p = commands.add_parser("run")
    p.add_argument("run")
    p.add_argument("--prefix", required=True, help="file name prefix, e.g. the experiment id")
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    if args.command == "splits":
        outputs = {"splits.svg": splits_figure()}
    elif args.command == "example":
        outputs = {"cer-example.svg": example_figure(args.run, args.dataset, args.sample_id)}
    elif args.command == "gallery":
        levels = [float(v) for v in args.levels.split(",")]
        outputs = {args.out: gallery_figure(args.run, args.dataset, levels)}
    else:
        runs = load_run(args.run)
        outputs = {
            f"{args.prefix}-cer-by-norm.svg": cer_by_norm_figure(runs),
            f"{args.prefix}-coverage.svg": coverage_figure(runs),
        }
    for name, content in outputs.items():
        (FIGURES / name).write_text(content, encoding="utf-8")
        print(FIGURES / name)


if __name__ == "__main__":
    main()
