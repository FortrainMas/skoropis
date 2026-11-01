"""Command line entry point: ``diploma {data,split,eval,compare,train,train-trocr}``."""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import importlib.util
import io
import platform
import random
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from diploma.data import (
    DATA_ROOT,
    DATASETS,
    REPO_ROOT,
    SPLITS,
    SPLITS_ROOT,
    load_dataset,
    make_group_split,
    makarov_pages,
    write_split,
)
from diploma.evaluation import evaluate, load_reports

RUNS_ROOT = REPO_ROOT / "runs"


def main(argv: Sequence[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="diploma", description="Handwritten text recognition experiments.")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("data", help="line and page counts per split, alphabets")
    p.add_argument("datasets", nargs="*", default=list(DATASETS), metavar="DATASET")
    p.set_defaults(run=cmd_data)

    p = commands.add_parser("split", help="create the page-level train/val/test split of makarov")
    p.add_argument("--val", type=float, default=0.15, help="fraction of pages (default: 0.15)")
    p.add_argument("--test", type=float, default=0.2, help="fraction of pages (default: 0.2)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="overwrite an existing split (changes the test set!)")
    p.set_defaults(run=cmd_split)

    p = commands.add_parser("eval", help="evaluate a recognizer on one or more datasets")
    p.add_argument("model", help="factory returning a Recognizer: 'package.module:Name' or 'path/to/file.py:Name'")
    p.add_argument("-a", "--arg", action="append", default=[], metavar="KEY=VALUE", help="factory keyword argument")
    p.add_argument("-d", "--data", action="append", required=True, metavar="DATASET[:SPLIT]")
    p.add_argument("--name", help="run name (default: <factory>-<timestamp>)")
    p.add_argument("--runs", type=Path, default=RUNS_ROOT, help="where runs are stored (default: runs/)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--sample", type=int, metavar="N", help="a random subset of N lines of each dataset")
    p.add_argument("--seed", type=int, default=0, help="seed of --sample (default: 0)")
    p.add_argument("--limit", type=int, help="only the first N samples of each dataset, for smoke tests")
    p.set_defaults(run=cmd_eval)

    p = commands.add_parser("train", help="train the CRNN-CTC baseline (see diploma.training)")
    p.add_argument("--name", required=True, help="checkpoints go to checkpoints/<name>/")
    p.add_argument("--train", action="append", required=True, metavar="DATASET[:SPLIT][*N]",
                   help="training set, repeated N times per epoch (repeatable)")
    p.add_argument("--val", action="append", required=True, metavar="DATASET[:SPLIT]",
                   help="validation set; the first one selects best.pt (repeatable)")
    p.add_argument("--init", help="checkpoint to fine-tune from")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--height", type=int)
    p.add_argument("--max-width", type=int)
    p.add_argument("--hidden", type=int)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=None)
    p.add_argument("--no-augment", dest="augment", action="store_false", default=None)
    p.add_argument("--workers", type=int)
    p.add_argument("--device", help="cuda, cpu or auto (default)")
    p.add_argument("--seed", type=int)
    p.add_argument("--limit", type=int, help="first N lines of every set, for smoke runs")
    p.add_argument("--out", type=Path, help="checkpoint root (default: checkpoints/)")
    p.set_defaults(run=cmd_train)

    p = commands.add_parser("train-trocr", help="fine-tune a TrOCR-style model (see diploma.trocr_training)")
    p.add_argument("--name", required=True, help="weights go to checkpoints/<name>/best/")
    p.add_argument("--train", action="append", required=True, metavar="DATASET[:SPLIT][*N]")
    p.add_argument("--val", action="append", required=True, metavar="DATASET[:SPLIT]",
                   help="validation set; the first one selects best/ (repeatable)")
    p.add_argument("--checkpoint", help="Hub id or local directory to start from (default: Kansallisarkisto)")
    p.add_argument("--revision")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--accumulate", type=int, help="batches per optimizer step")
    p.add_argument("--lr", type=float)
    p.add_argument("--freeze-encoder", action="store_true", default=None)
    p.add_argument("--freeze-decoder-epochs", type=int, help="train only the encoder for the first N epochs")
    p.add_argument("--no-augment", dest="augment", action="store_false", default=None)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false", default=None)
    p.add_argument("--workers", type=int)
    p.add_argument("--device", help="cuda, cpu or auto (default)")
    p.add_argument("--seed", type=int)
    p.add_argument("--limit", type=int, help="first N lines of every set, for smoke runs")
    p.add_argument("--out", type=Path, help="checkpoint root (default: checkpoints/)")
    p.set_defaults(run=cmd_train_trocr)

    p = commands.add_parser("compare", help="table of every evaluated run")
    p.add_argument("--runs", type=Path, default=RUNS_ROOT)
    p.add_argument("-d", "--data", metavar="DATASET[:SPLIT]", help="only this dataset")
    p.set_defaults(run=cmd_compare)

    args = parser.parse_args(argv)
    args.run(args)


def cmd_data(args: argparse.Namespace) -> None:
    rows = []
    for name in args.datasets:
        samples = load_dataset(name)
        for split in SPLITS:
            part = [s for s in samples if s.split == split]
            chars = sum(len(s.text) for s in part)
            rows.append([name, split, len(part), len({s.group for s in part}), chars])
        alphabet = Counter(c for s in samples for c in s.text)
        rows.append([name, "all", len(samples), len({s.group for s in samples}), alphabet.total()])
        print(f"{name}: {len(alphabet)} symbols: {''.join(sorted(alphabet))}")
    print()
    print(_markdown(["dataset", "split", "lines", "pages", "chars"], rows))


def cmd_split(args: argparse.Namespace) -> None:
    path = SPLITS_ROOT / "makarov.json"
    if path.exists() and not args.force:
        sys.exit(f"{path} already exists; pass --force to overwrite it (this changes the test set)")
    split = make_group_split(makarov_pages(DATA_ROOT), {"test": args.test, "val": args.val}, args.seed)
    write_split(path, split, seed=args.seed)
    print(f"wrote {path}: " + ", ".join(f"{name} {len(pages)} pages" for name, pages in split.items()))


def cmd_eval(args: argparse.Namespace) -> None:
    datasets = {}
    for spec in args.data:
        samples = load_dataset(spec)
        if args.sample is not None and args.sample < len(samples):
            samples = sorted(random.Random(args.seed).sample(samples, args.sample), key=lambda s: s.id)
        datasets[spec] = samples[: args.limit]
        if not datasets[spec]:
            sys.exit(f"{spec}: no samples")

    kwargs = dict(_parse_kwarg(item) for item in args.arg)
    recognizer = load_object(args.model)(**kwargs)
    describe = getattr(recognizer, "describe", None)
    name = args.name or f"{args.model.rpartition(':')[2].lower()}-{datetime.now():%Y%m%d-%H%M%S}"

    reports = []
    for spec, samples in datasets.items():
        meta = {
            "run": name,
            "model": args.model,
            "model_args": kwargs,
            "model_info": describe() if callable(describe) else None,
            "dataset": spec,
            "sample": args.sample,
            "seed": args.seed if args.sample is not None else None,
            "limit": args.limit,
            "batch_size": args.batch_size,
            "created": datetime.now().isoformat(timespec="seconds"),
            "environment": environment(),
        }
        out_dir = args.runs / name / spec.replace(":", "_")
        reports.append(
            evaluate(recognizer, samples, out_dir, batch_size=args.batch_size, progress=_progress(spec), meta=meta)
        )
    print(format_reports(reports))
    print(f"\nsaved to {args.runs / name}")


def cmd_train(args: argparse.Namespace) -> None:
    from diploma.training import TrainConfig, train  # torch-heavy, keep other commands fast

    options = {k: v for k, v in vars(args).items() if k not in ("command", "run") and v is not None}
    print(f"best checkpoint: {train(TrainConfig(**options))}")


def cmd_train_trocr(args: argparse.Namespace) -> None:
    from diploma.trocr_training import TrOCRTrainConfig, train_trocr  # transformers-heavy

    options = {k: v for k, v in vars(args).items() if k not in ("command", "run") and v is not None}
    print(f"best weights: {train_trocr(TrOCRTrainConfig(**options))}")


def cmd_compare(args: argparse.Namespace) -> None:
    reports = [r for r in load_reports(args.runs) if args.data in (None, r.get("dataset"))]
    if not reports:
        sys.exit(f"no runs under {args.runs}")
    print(format_reports(sorted(reports, key=lambda r: (r["dataset"], r["metrics"]["exact"]["cer"]))))


def format_reports(reports: Sequence[dict[str, Any]]) -> str:
    header = ["run", "dataset", "lines", "CER %", "95% CI", "WER %", "line acc %",
              "CER caseless", "CER letters", "CER modern", "lines/s"]  # fmt: skip
    rows = []
    for r in reports:
        m = r["metrics"]
        low, high = m["exact"]["ci95"]["cer"]
        dataset = r["dataset"] + (f" [sample {r['sample']}]" if r.get("sample") else "")
        dataset += f" [first {r['limit']}]" if r.get("limit") else ""
        speed = r.get("lines_per_second")
        rows.append([
            r["run"], dataset, r["lines"],
            _pct(m["exact"]["cer"]), f"{_pct(low)}-{_pct(high)}", _pct(m["exact"]["wer"]),
            _pct(m["exact"]["line_accuracy"]),
            _pct(m["caseless"]["cer"]), _pct(m["letters"]["cer"]), _pct(m["modern"]["cer"]),
            f"{speed:.1f}" if speed else "-",
        ])  # fmt: skip
    return _markdown(header, rows)


def environment() -> dict[str, Any]:
    """What is needed to tell whether two runs are comparable."""
    packages = {}
    for package in ("torch", "transformers"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        git = ["git", "-C", str(REPO_ROOT)]
        commit = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run([*git, "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
        "git_commit": commit,
        "git_dirty": dirty,
    }


def load_object(spec: str) -> Any:
    """Resolve ``module:attr`` or ``path/to/file.py:attr``."""
    target, sep, attr = spec.rpartition(":")
    if not sep or not target or not attr:
        raise ValueError(f"expected 'module:attr' or 'file.py:attr', got {spec!r}")
    if target.endswith(".py"):
        module_spec = importlib.util.spec_from_file_location(Path(target).stem, target)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot load {target}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, attr)


def _parse_kwarg(item: str) -> tuple[str, Any]:
    key, sep, raw = item.partition("=")
    if not sep:
        raise ValueError(f"expected KEY=VALUE, got {item!r}")
    try:
        return key, ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return key, raw


def _progress(label: str):
    if not sys.stderr.isatty():
        return None

    def report(done: int, total: int) -> None:
        print(f"\r{label}: {done}/{total}", end="" if done < total else "\n", file=sys.stderr, flush=True)

    return report


def _pct(value: float) -> str:
    return f"{100 * value:.2f}"


def _markdown(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    cells = [list(map(str, header))] + [list(map(str, row)) for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
    lines = ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in cells]
    lines.insert(1, "|" + "|".join("-" * (w + 2) for w in widths) + "|")
    return "\n".join(lines)
