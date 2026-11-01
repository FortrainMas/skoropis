"""Fine-tuning of a TrOCR-style encoder-decoder (by default Kansallisarkisto's) on our lines.

    uv run diploma train-trocr --name trocr-makarov --train makarov:train --val makarov:val

Labels follow the checkpoint's own convention, ``<s> text </s>`` (its generations
start with the decoder start token followed by ``<s>``), so fine-tuning continues
the task the model was trained on instead of teaching it a new format.

Every epoch the model reads the validation lines greedily through the same
evaluation code as ``diploma eval``; the weights with the lowest CER on the first
validation set are saved with ``save_pretrained`` as ``best/`` and can be
evaluated with ``diploma eval diploma.models.trocr:TrOCR -a checkpoint=.../best``.
Validation is greedy for speed; the final evaluation uses the checkpoint's beam
search, so test numbers are usually a little better than the logged ones.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import VisionEncoderDecoderModel

from diploma.data import Sample, load_dataset
from diploma.models.trocr import DEFAULT_CHECKPOINT, TrOCR, _Processor
from diploma.training import CHECKPOINTS, augment, jsonable, load_training_set, mixed_precision, validate


@dataclass
class TrOCRTrainConfig:
    name: str
    train: list[str]  # dataset specs; "makarov:train*4" repeats the set 4 times per epoch
    val: list[str]  # the first one selects the best checkpoint
    checkpoint: str = DEFAULT_CHECKPOINT  # Hub id or a local directory saved by this script
    revision: str | None = None
    epochs: int = 20
    batch_size: int = 8
    accumulate: int = 2  # gradient accumulation: an optimizer step every `accumulate` batches
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup: float = 0.1  # share of optimizer steps with linearly growing learning rate
    freeze_encoder: bool = False
    freeze_decoder_epochs: int = 0  # the authors' recipe: let the encoder adapt first, keep the language model
    augment: bool = True
    gradient_checkpointing: bool = True
    workers: int = min(8, os.cpu_count() or 1)
    device: str = "auto"
    seed: int = 0
    limit: int | None = None  # first N lines of every set, for smoke runs
    out: Path = field(default=CHECKPOINTS)


class LineImages(Dataset):
    def __init__(self, samples: list[Sample], augment: bool) -> None:
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[Image.Image, str]:
        sample = self.samples[i]
        with Image.open(sample.image) as image:
            image = image.convert("RGB")
        return (augment(image) if self.augment else image), sample.text


class Collate:
    """Images to pixel values, texts to ``<s> text </s>`` token ids with padding masked as -100."""

    def __init__(self, processor: _Processor) -> None:
        self.processor = processor

    def __call__(self, batch: list[tuple[Image.Image, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        images, texts = zip(*batch)
        pixel_values = self.processor(images=list(images), return_tensors="pt").pixel_values
        tokens = self.processor.tokenizer(list(texts), padding=True, return_tensors="pt")
        labels = tokens.input_ids.masked_fill(tokens.attention_mask == 0, -100)
        return pixel_values, labels


def train_trocr(config: TrOCRTrainConfig) -> Path:
    """Fine-tune, validate every epoch, and return the directory of the best weights."""
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else
                          "cpu" if config.device == "auto" else config.device)  # fmt: skip
    out_dir = config.out / config.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(jsonable(config), ensure_ascii=False, indent=2), encoding="utf-8")

    processor = _Processor.from_pretrained(config.checkpoint, revision=config.revision)
    model = VisionEncoderDecoderModel.from_pretrained(config.checkpoint, revision=config.revision).to(device)
    if config.gradient_checkpointing:
        # Non-reentrant checkpointing keeps gradients right when a whole half of the model is frozen.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.encoder.requires_grad_(not config.freeze_encoder)

    train_samples = load_training_set(config.train, config.limit)
    val_sets = {spec: load_dataset(spec)[: config.limit] for spec in config.val}
    loader = DataLoader(
        LineImages(train_samples, config.augment),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=Collate(processor),
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.workers > 0,
    )
    steps = config.epochs * math.ceil(len(loader) / config.accumulate)
    warmup = max(1, round(config.warmup * steps))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup, max(0.0, (steps - step) / max(1, steps - warmup)))
    )
    amp_dtype, scaler = mixed_precision(device)
    recognizer = TrOCR.wrap(model, processor, device, num_beams=1, max_length=96)

    print(f"{len(train_samples)} training lines, {steps} optimizer steps, device {device}")
    best = float("inf")
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        tic = time.perf_counter()
        model.train()
        model.decoder.requires_grad_(epoch > config.freeze_decoder_epochs)
        losses = []
        for i, (pixel_values, labels) in enumerate(loader, 1):
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                loss = model(pixel_values=pixel_values.to(device), labels=labels.to(device)).loss
            scaler.scale(loss / config.accumulate).backward()
            losses.append(loss.item())
            if i % config.accumulate == 0 or i == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        model.eval()
        val = {spec: validate(recognizer, samples, config.batch_size) for spec, samples in val_sets.items()}
        score = val[config.val[0]]["cer"]
        entry = {"epoch": epoch, "loss": sum(losses) / len(losses), "lr": scheduler.get_last_lr()[0],
                 "val": val, "seconds": time.perf_counter() - tic}  # fmt: skip
        with open(out_dir / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        improved = score < best
        if improved:
            best = score
            model.save_pretrained(out_dir / "best")
            processor.save_pretrained(out_dir / "best")
        scores = "  ".join(f"{spec} CER {v['cer'] * 100:.2f}%" for spec, v in val.items())
        left = (time.perf_counter() - started) / epoch * (config.epochs - epoch)
        print(f"epoch {epoch:3d}  loss {entry['loss']:.4f}  {scores}  {entry['seconds']:.0f}s  "
              f"left ~{left / 60:.0f} min" + ("  *" if improved else ""), flush=True)  # fmt: skip
    return out_dir / "best"

