"""Training of the CRNN-CTC baseline.

    uv run diploma train --name crnn-peter --train peter:train --val peter:val
    uv run diploma train --name crnn-makarov --init checkpoints/crnn-peter/best.pt \\
        --train makarov:train --val makarov:val

Every epoch the current weights are scored on the validation sets with the same
code as ``diploma eval`` (greedy decoding), and the checkpoint with the lowest
CER on the first validation set is kept as ``best.pt``. Test sets are never
touched here.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import v2

from diploma.data import REPO_ROOT, Sample, load_dataset
from diploma.evaluation import Recognizer, predict
from diploma.metrics import Tally, total
from diploma.models.crnn import CRNN, Alphabet, CRNNRecognizer, pad_batch, to_tensor
from diploma.text import Norm, normalize

CHECKPOINTS = REPO_ROOT / "checkpoints"


@dataclass
class TrainConfig:
    name: str
    train: list[str]  # dataset specs; "makarov:train*4" repeats the set 4 times per epoch
    val: list[str]  # the first one selects the best checkpoint
    init: str | None = None  # checkpoint to fine-tune; its architecture wins over height/hidden
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-2
    height: int = 128
    max_width: int = 2048
    hidden: int = 256
    dropout: float = 0.2
    pretrained: bool = True  # ImageNet weights for the ResNet stem
    augment: bool = True
    workers: int = min(8, os.cpu_count() or 1)  # image decoding and augmentation processes
    device: str = "auto"
    seed: int = 0
    limit: int | None = None  # first N lines of every set, for smoke runs
    out: Path = field(default=CHECKPOINTS)


# --- data --------------------------------------------------------------------

_AUGMENT = v2.Compose([
    v2.RandomApply([v2.RandomAffine(degrees=2, shear=(-12, 12, -2, 2), scale=(0.9, 1.0), fill=255)], p=0.7),
    v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
    v2.RandomApply([v2.GaussianBlur(3, sigma=(0.1, 1.2))], p=0.2),
])  # fmt: skip


def augment(image: Image.Image) -> Image.Image:
    """Slant, small rotation and scale, ink/paper contrast, blur, and horizontal stretch."""
    stretch = float(torch.empty(1).uniform_(0.8, 1.2))  # torch RNG: seeded per DataLoader worker
    image = image.resize((max(1, round(image.width * stretch)), image.height), Image.BILINEAR)
    return _AUGMENT(image)


class LineDataset(Dataset):
    def __init__(self, samples: list[Sample], alphabet: Alphabet, height: int, max_width: int, augment: bool) -> None:
        self.samples = samples
        self.alphabet = alphabet
        self.height = height
        self.max_width = max_width
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[i]
        with Image.open(sample.image) as image:
            image = image.convert("L")
        if self.augment:
            image = augment(image)
        return to_tensor(image, self.height, self.max_width), torch.tensor(self.alphabet.encode(sample.text))


def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, ...]:
    lines, targets = zip(*batch)
    images, widths = pad_batch(lines)
    return images, widths, torch.cat(targets), torch.tensor([len(t) for t in targets])


class WidthBuckets(Sampler[list[int]]):
    """Random batches of lines with similar aspect ratio.

    Lines are shuffled, cut into pools of ``pool`` batches, sorted by width inside
    a pool and batched; the batches are then shuffled again. A batch is padded to
    its widest line, so this saves the compute spent on white padding (about a
    third on Digital Peter) while keeping every epoch random.
    """

    def __init__(self, widths: list[float], batch_size: int, pool: int = 50) -> None:
        self.widths = widths
        self.batch_size = batch_size
        self.pool_size = batch_size * pool

    def __iter__(self):
        order = torch.randperm(len(self.widths)).tolist()
        batches = []
        for start in range(0, len(order), self.pool_size):
            pool = sorted(order[start : start + self.pool_size], key=self.widths.__getitem__)
            batches += [pool[i : i + self.batch_size] for i in range(0, len(pool), self.batch_size)]
        for k in torch.randperm(len(batches)).tolist():
            yield batches[k]

    def __len__(self) -> int:
        full, rest = divmod(len(self.widths), self.pool_size)
        return full * -(-self.pool_size // self.batch_size) + -(-rest // self.batch_size)


def aspect_ratios(samples: list[Sample]) -> list[float]:
    """Width / height of every line image, read from file headers only."""
    cache: dict[Path, float] = {}
    for sample in samples:
        if sample.image not in cache:
            with Image.open(sample.image) as image:
                cache[sample.image] = image.width / image.height
    return [cache[s.image] for s in samples]


def load_training_set(specs: list[str], limit: int | None) -> list[Sample]:
    samples = []
    for spec in specs:
        name, _, repeat = spec.partition("*")
        samples += load_dataset(name)[:limit] * int(repeat or 1)
    return samples


# --- training ----------------------------------------------------------------


def train(config: TrainConfig) -> Path:
    """Train, validate every epoch, and return the path of the best checkpoint."""
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else
                          "cpu" if config.device == "auto" else config.device)  # fmt: skip
    out_dir = config.out / config.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(jsonable(config), ensure_ascii=False, indent=2), encoding="utf-8")

    train_samples = load_training_set(config.train, config.limit)
    val_sets = {spec: load_dataset(spec)[: config.limit] for spec in config.val}
    texts = [s.text for s in train_samples]
    if config.init:
        model, alphabet = _from_checkpoint(config.init, texts)
    else:
        alphabet = Alphabet.from_texts(texts)
        model = CRNN(alphabet.num_classes, config.height, config.hidden, config.dropout, config.pretrained)
    model.to(device)
    height = model.hparams["height"]

    loader = DataLoader(
        LineDataset(train_samples, alphabet, height, config.max_width, config.augment),
        batch_sampler=WidthBuckets(aspect_ratios(train_samples), config.batch_size),
        collate_fn=collate,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.workers > 0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.lr, total_steps=config.epochs * len(loader), pct_start=0.1
    )
    amp_dtype, scaler = mixed_precision(device)
    ctc =nn.CTCLoss(blank=0, zero_infinity=True)
    recognizer = CRNNRecognizer(model, alphabet, config.max_width, device)

    print(f"{len(train_samples)} training lines, {alphabet.num_classes - 1} symbols, device {device}")
    best = float("inf")
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        tic = time.perf_counter()
        model.train()
        losses = []
        for images, widths, targets, target_lengths in loader:
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                log_probs, lengths = model(images.to(device, non_blocking=True), widths.to(device))
            loss = ctc(log_probs.transpose(0, 1), targets.to(device), lengths, target_lengths.to(device))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())

        model.eval()
        val = {spec: validate(recognizer, samples, config.batch_size) for spec, samples in val_sets.items()}
        score = val[config.val[0]]["cer"]
        entry = {"epoch": epoch, "loss": sum(losses) / len(losses), "lr": scheduler.get_last_lr()[0],
                 "val": val, "seconds": time.perf_counter() - tic}  # fmt: skip
        with open(out_dir / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        state = {
            "model": model.state_dict(),
            "hparams": model.hparams,
            "alphabet": alphabet.chars,
            "max_width": config.max_width,
            "epoch": epoch,
            "val": val,
            "config": jsonable(config),
        }
        torch.save(state, out_dir / "last.pt")
        improved = score < best
        if improved:
            best = score
            torch.save(state, out_dir / "best.pt")
        scores = "  ".join(f"{spec} CER {v['cer'] * 100:.2f}%" for spec, v in val.items())
        left = (time.perf_counter() - started) / epoch * (config.epochs - epoch)
        print(f"epoch {epoch:3d}  loss {entry['loss']:.4f}  {scores}  {entry['seconds']:.0f}s  "
              f"left ~{left / 60:.0f} min" + ("  *" if improved else ""), flush=True)  # fmt: skip
    return out_dir / "best.pt"


def validate(recognizer: Recognizer, samples: list[Sample], batch_size: int) -> dict[str, float]:
    """Point estimates only: intervals and confusions are for ``diploma eval``, not for every epoch."""
    records, _ = predict(recognizer, samples, batch_size=batch_size)
    exact, modern = (
        total(Tally.of_line(normalize(r.ref, norm), normalize(r.hyp, norm)) for r in records)
        for norm in (Norm.EXACT, Norm.MODERN)
    )
    return {"cer": exact.cer, "wer": exact.wer, "line_accuracy": exact.line_accuracy, "cer_modern": modern.cer}


def mixed_precision(device: torch.device) -> tuple[torch.dtype, torch.amp.GradScaler]:
    """Autocast dtype and loss scaler: bf16 where the GPU has it in hardware (Ampere and newer), else fp16.

    ``torch.cuda.is_bf16_supported()`` on its own also says yes when bf16 is only emulated
    (T4 and other pre-Ampere cards), and emulated bf16 is several times slower than fp16 there.
    """
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda" and not bf16)
    return (torch.bfloat16 if bf16 else torch.float16), scaler


def _from_checkpoint(path: str, texts: list[str]) -> tuple[CRNN, Alphabet]:
    """Weights of ``path``; the output layer grows by the symbols ``texts`` add to its alphabet."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    alphabet = Alphabet(state["alphabet"]).extended(texts)
    model = CRNN(alphabet.num_classes, **state["hparams"])
    weights = dict(state["model"])
    head_weight, head_bias = weights.pop("head.weight"), weights.pop("head.bias")
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if set(missing) != {"head.weight", "head.bias"} or unexpected:
        raise ValueError(f"{path} does not match the CRNN architecture: missing {missing}, unexpected {unexpected}")
    with torch.no_grad():
        model.head.weight[: len(head_weight)] = head_weight
        model.head.bias[: len(head_bias)] = head_bias
    return model, alphabet


def jsonable(config: Any) -> dict[str, Any]:
    return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}
