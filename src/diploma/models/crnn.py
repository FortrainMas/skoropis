"""CRNN baseline: ResNet34 feature extractor, BiLSTM, CTC.

The shape of the strong Digital Peter solutions (AIJ 2020 winners, StackMix):
the first three ResNet34 stages, with the last two downsampling only height,
so one output frame covers 4 pixels of the line and even narrow letters get
several frames. Remaining height is flattened into channels, so the sequence
model still sees where in the line height a stroke is.

Input lines are converted to grayscale, resized to a fixed height keeping the
aspect ratio and right-padded with white within a batch.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.transforms.functional import pil_to_tensor

from diploma.evaluation import Prediction

BLANK = 0


@dataclass(frozen=True)
class Alphabet:
    """Symbol ``chars[i]`` is class ``i + 1``; class 0 is the CTC blank."""

    chars: str

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> Alphabet:
        return cls("".join(sorted(set("".join(texts)))))

    def extended(self, texts: Iterable[str]) -> Alphabet:
        """This alphabet plus unseen symbols of ``texts``, appended so existing classes keep their ids."""
        new = sorted(set("".join(texts)) - set(self.chars))
        return Alphabet(self.chars + "".join(new))

    @cached_property
    def _ids(self) -> dict[str, int]:
        return {c: i + 1 for i, c in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        return [self._ids[c] for c in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.chars[i - 1] for i in ids)

    @property
    def num_classes(self) -> int:
        return len(self.chars) + 1


# --- preprocessing ---------------------------------------------------------


def to_tensor(image: Image.Image, height: int, max_width: int) -> torch.Tensor:
    """Grayscale ``(1, height, width)`` tensor in [0, 1], white = 1."""
    gray = image.convert("L")
    width = max(4, min(max_width, round(gray.width * height / gray.height)))
    return pil_to_tensor(gray.resize((width, height), Image.BILINEAR)).float().div_(255)


def pad_batch(lines: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad with white to the widest line; returns the batch and the true widths."""
    widths = torch.tensor([line.shape[-1] for line in lines])
    batch = torch.ones(len(lines), *lines[0].shape[:-1], int(widths.max()))
    for i, line in enumerate(lines):
        batch[i, ..., : line.shape[-1]] = line
    return batch, widths


def output_length(widths: torch.Tensor) -> torch.Tensor:
    """Number of CTC frames for input widths: two stride-2 layers, each rounding up."""
    return (widths + 3) // 4


# --- model -----------------------------------------------------------------


class CRNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        height: int = 128,
        hidden: int = 256,
        dropout: float = 0.2,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if height % 16:
            raise ValueError(f"height must be divisible by 16, got {height}")
        self.hparams = {"height": height, "hidden": hidden, "dropout": dropout}
        resnet = torchvision.models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        for stage in (resnet.layer2, resnet.layer3):
            stage[0].conv1.stride = (2, 1)
            stage[0].downsample[0].stride = (2, 1)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3
        )
        self.project = nn.Sequential(nn.Linear(256 * (height // 16), hidden), nn.Dropout(dropout))
        self.rnn = nn.LSTM(hidden, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)
        self.head = nn.Linear(2 * hidden, num_classes)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, lines: torch.Tensor, widths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(N, 1, H, W)`` lines -> ``(N, T, classes)`` log-probabilities and valid frame counts."""
        x = (lines.expand(-1, 3, -1, -1) - self.mean) / self.std  # ImageNet statistics for the ResNet stem
        x = self.backbone(x)
        n, c, h, t = x.shape
        x = self.project(x.permute(0, 3, 1, 2).reshape(n, t, c * h))
        lengths = output_length(widths).clamp(max=t)
        # The recurrent part and the head run in float32 even under autocast: they are a few
        # percent of the compute, and cuDNN's half-precision LSTM support varies across versions.
        with torch.autocast(x.device.type, enabled=False):
            packed = pack_padded_sequence(x.float(), lengths.cpu(), batch_first=True, enforce_sorted=False)
            x, _ = pad_packed_sequence(self.rnn(packed)[0], batch_first=True, total_length=t)
            return self.head(x).log_softmax(-1), lengths


def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor) -> list[tuple[list[int], float]]:
    """Best path per line: collapse repeats, drop blanks.

    Confidence is the geometric mean of the per-frame best-path probability.
    """
    best, ids = log_probs.max(-1)
    decoded = []
    for frames, scores, length in zip(ids.tolist(), best, lengths.tolist()):
        frames = frames[:length]
        symbols = [s for k, s in enumerate(frames) if s != BLANK and (k == 0 or s != frames[k - 1])]
        decoded.append((symbols, scores[:length].mean().exp().item()))
    return decoded


# --- recognizer --------------------------------------------------------------


class CRNNRecognizer:
    """:class:`diploma.evaluation.Recognizer` around a trained :class:`CRNN`."""

    def __init__(
        self,
        model: CRNN,
        alphabet: Alphabet,
        max_width: int,
        device: str | torch.device = "cpu",
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.alphabet = alphabet
        self.max_width = max_width
        self.device = torch.device(device)
        self.meta = meta or {}

    @classmethod
    def load(cls, checkpoint: str | Path, device: str = "cpu") -> CRNNRecognizer:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        alphabet = Alphabet(state["alphabet"])
        model = CRNN(alphabet.num_classes, **state["hparams"])
        model.load_state_dict(state["model"])
        meta = {"checkpoint": str(checkpoint), "epoch": state["epoch"], "val": state["val"]}
        return cls(model, alphabet, state["max_width"], device, meta)

    def describe(self) -> dict[str, Any]:
        return {
            **self.meta,
            "hparams": self.model.hparams,
            "max_width": self.max_width,
            "alphabet_size": len(self.alphabet.chars),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "decoding": "greedy CTC",
        }

    @torch.inference_mode()
    def predict(self, images: Sequence[Image.Image]) -> list[Prediction]:
        lines = [to_tensor(image, self.model.hparams["height"], self.max_width) for image in images]
        batch, widths = pad_batch(lines)
        log_probs, lengths = self.model(batch.to(self.device), widths.to(self.device))
        return [Prediction(self.alphabet.decode(ids), confidence) for ids, confidence in greedy_decode(log_probs, lengths)]


def load(checkpoint: str, device: str = "cpu") -> CRNNRecognizer:
    """Factory for ``diploma eval diploma.models.crnn:load -a checkpoint=...``."""
    return CRNNRecognizer.load(checkpoint, device)
