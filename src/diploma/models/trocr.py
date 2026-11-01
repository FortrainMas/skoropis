"""TrOCR-style Hugging Face ``VisionEncoderDecoderModel`` checkpoints as a :class:`Recognizer`.

Generation follows the checkpoint's own ``generation_config`` (beam size, n-gram
blocking, length penalty) unless overridden, so a zero-shot run measures the
model as its authors ship it.

Confidence is the geometric mean of the chosen tokens' probabilities, i.e.
``exp(mean log p)`` over the generated sequence.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import try_to_load_from_cache
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from diploma.evaluation import Prediction

DEFAULT_CHECKPOINT = "Kansallisarkisto/cyrillic-large-handwritten"


class _Processor(TrOCRProcessor):
    # The Kansallisarkisto checkpoints pair a BiT image processor with a RoBERTa
    # tokenizer, which TrOCRProcessor's type checks reject; their model card
    # loads them through this same pass-through subclass.
    def __init__(self, image_processor, tokenizer) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.current_processor = image_processor
        self.chat_template = None


class TrOCR:
    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        revision: str | None = None,
        device: str = "cpu",
        **generation: Any,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.processor = _Processor.from_pretrained(checkpoint, revision=revision)
        self.model = VisionEncoderDecoderModel.from_pretrained(checkpoint, revision=revision).to(self.device).eval()
        self.revision = revision or _cached_revision(checkpoint)
        # Several checkpoints ship use_cache=False, which makes decoding quadratic for no benefit.
        self.generation = {"use_cache": True, **generation}

    @classmethod
    def wrap(
        cls, model: VisionEncoderDecoderModel, processor: _Processor, device: str | torch.device, **generation: Any
    ) -> TrOCR:
        """A recognizer around an already loaded model, e.g. the one being fine-tuned."""
        self = cls.__new__(cls)
        self.checkpoint, self.revision, self.device = "<in memory>", None, torch.device(device)
        self.processor, self.model = processor, model
        self.generation = {"use_cache": True, **generation}
        return self

    def describe(self) -> dict[str, Any]:
        config = self.model.generation_config.to_dict() | self.generation
        return {
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "encoder": self.model.config.encoder.model_type,
            "decoder": self.model.config.decoder.model_type,
            "generation": {
                key: config.get(key)
                for key in ("num_beams", "max_length", "no_repeat_ngram_size", "length_penalty", "do_sample")
            },
        }

    @torch.inference_mode()
    def predict(self, images: Sequence[Image.Image]) -> list[Prediction]:
        pixel_values = self.processor(images=list(images), return_tensors="pt").pixel_values.to(self.device)
        out = self.model.generate(
            pixel_values, output_scores=True, return_dict_in_generate=True, **self.generation
        )
        texts = self.processor.batch_decode(out.sequences, skip_special_tokens=True)
        return [Prediction(text, confidence) for text, confidence in zip(texts, self._confidences(out))]

    def _confidences(self, out: Any) -> list[float]:
        beam_indices = getattr(out, "beam_indices", None)
        # Beam search already reports log-probabilities; greedy reports raw logits.
        log_probs = self.model.compute_transition_scores(
            out.sequences, out.scores, beam_indices, normalize_logits=beam_indices is None
        )
        generated = out.sequences[:, 1 : 1 + log_probs.shape[1]]  # drop decoder_start_token
        mask = generated != self.model.generation_config.pad_token_id
        mean = (log_probs.masked_fill(~mask, 0).sum(1) / mask.sum(1).clamp(min=1)).exp()
        return mean.tolist()


def _cached_revision(checkpoint: str) -> str | None:
    """Commit hash of the Hub snapshot that was loaded; ``None`` for local directories."""
    if Path(checkpoint).is_dir():
        return None
    path = try_to_load_from_cache(checkpoint, "config.json")
    return Path(path).parent.name if isinstance(path, str) else None
