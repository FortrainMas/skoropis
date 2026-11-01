"""Fine-tuning on a tiny, randomly initialized checkpoint built from the same parts as Kansallisarkisto's."""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing
from transformers import (
    PreTrainedTokenizerFast,
    RobertaConfig,
    ViTConfig,
    ViTImageProcessor,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)

from diploma.data import load_dataset
from diploma.evaluation import evaluate
from diploma.models.trocr import TrOCR, _Processor
from diploma.trocr_training import Collate, TrOCRTrainConfig, train_trocr

PAD, BOS, EOS = 0, 1, 2


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> Path:
    """A saved checkpoint: ViT encoder, RoBERTa decoder, byte-level BPE with ``<s> text </s>`` labels."""
    texts = ["Рѣшенiе объявлено", "судъ", "№ 82.", "сторонамъ", "катеринушка", "друг мой"] * 10
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(texts, vocab_size=300, special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"])
    bpe.post_processor = TemplateProcessing(single="<s> $A </s>", special_tokens=[("<s>", BOS), ("</s>", EOS)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=bpe._tokenizer, pad_token="<pad>", bos_token="<s>", eos_token="</s>", unk_token="<unk>"
    )
    processor = _Processor(ViTImageProcessor(size={"height": 32, "width": 128}), tokenizer)

    encoder = ViTConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                        image_size=(32, 128), patch_size=16)  # fmt: skip
    decoder = RobertaConfig(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                            intermediate_size=64, is_decoder=True, add_cross_attention=True, pad_token_id=PAD,
                            bos_token_id=BOS, eos_token_id=EOS)  # fmt: skip
    torch.manual_seed(0)
    model = VisionEncoderDecoderModel(VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder, decoder))
    for cfg in (model.config, model.generation_config):
        cfg.decoder_start_token_id, cfg.pad_token_id, cfg.eos_token_id = BOS, PAD, EOS
    model.generation_config.max_length = 24

    path = tmp_path_factory.mktemp("tiny")
    model.save_pretrained(path)
    processor.save_pretrained(path)
    return path


@pytest.fixture
def data(roots: tuple[Path, Path], monkeypatch) -> None:
    monkeypatch.setattr("diploma.data.DATA_ROOT", roots[0])
    monkeypatch.setattr("diploma.data.SPLITS_ROOT", roots[1])


def config(tmp_path: Path, checkpoint: Path, **overrides) -> TrOCRTrainConfig:
    options = dict(name="tiny", train=["makarov:train*2"], val=["makarov:test"], checkpoint=str(checkpoint),
                   epochs=2, batch_size=2, accumulate=2, lr=1e-3, workers=0, device="cpu", out=tmp_path)  # fmt: skip
    return TrOCRTrainConfig(**(options | overrides))


def test_labels_are_wrapped_and_padding_is_ignored(tiny_checkpoint: Path) -> None:
    processor = _Processor.from_pretrained(tiny_checkpoint)
    image = Image.new("RGB", (100, 20), "white")
    _, labels = Collate(processor)([(image, "судъ"), (image, "Рѣшенiе объявлено")])
    assert labels[:, 0].tolist() == [BOS, BOS]
    assert (labels[0] == -100).any() and not (labels[1] == -100).any()
    assert labels[1, -1].item() == EOS
    assert processor.tokenizer.decode(labels[1], skip_special_tokens=True) == "Рѣшенiе объявлено"


def test_fine_tuning_saves_a_checkpoint_that_evaluates(tiny_checkpoint: Path, tmp_path: Path, data) -> None:
    best = train_trocr(config(tmp_path, tiny_checkpoint))
    history = [json.loads(l) for l in (tmp_path / "tiny" / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [h["epoch"] for h in history] == [1, 2]
    assert 0 <= history[-1]["val"]["makarov:test"]["cer"]

    recognizer = TrOCR(checkpoint=str(best))
    assert recognizer.describe()["revision"] is None
    report = evaluate(recognizer, load_dataset("makarov:test"))
    assert report["lines"] == 1 and "coverage" in report


@pytest.mark.parametrize(
    ("overrides", "frozen", "trained"),
    [({"freeze_encoder": True}, "encoder", "decoder"), ({"freeze_decoder_epochs": 1}, "decoder", "encoder")],
)
def test_freezing(tiny_checkpoint: Path, tmp_path: Path, data, overrides: dict, frozen: str, trained: str) -> None:
    best = train_trocr(config(tmp_path, tiny_checkpoint, epochs=1, **overrides))
    before = VisionEncoderDecoderModel.from_pretrained(tiny_checkpoint)
    after = VisionEncoderDecoderModel.from_pretrained(best)

    def same(part: str) -> bool:
        return all(torch.equal(a, b) for a, b in zip(getattr(before, part).parameters(), getattr(after, part).parameters()))

    assert same(frozen) and not same(trained)
