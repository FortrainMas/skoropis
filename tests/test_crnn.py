import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from diploma.cli import main
from diploma.data import load_dataset
from diploma.evaluation import evaluate
from diploma.models.crnn import CRNN, Alphabet, CRNNRecognizer, greedy_decode, output_length, pad_batch, to_tensor
from diploma.training import TrainConfig, mixed_precision, train


def test_alphabet_roundtrip_and_extension() -> None:
    alphabet = Alphabet.from_texts(["судъ", "дѣло"])
    assert alphabet.chars == "длосуъѣ"
    assert alphabet.decode(alphabet.encode("дѣло")) == "дѣло"
    assert alphabet.num_classes == 8
    with pytest.raises(KeyError):
        alphabet.encode("Пётр")
    extended = alphabet.extended(["судья"])
    assert extended.chars[: len(alphabet.chars)] == alphabet.chars
    assert extended.encode("дѣло") == alphabet.encode("дѣло")


def test_preprocessing_keeps_aspect_and_pads_white() -> None:
    line = to_tensor(Image.new("RGB", (300, 60), "black"), height=32, max_width=1000)
    assert line.shape == (1, 32, 160) and line.max() == 0
    assert to_tensor(Image.new("RGB", (3000, 30), "black"), height=32, max_width=1000).shape[-1] == 1000
    batch, widths = pad_batch([line, line[..., :100]])
    assert batch.shape == (2, 1, 32, 160) and widths.tolist() == [160, 100]
    assert batch[1, ..., 100:].min() == 1.0


@pytest.mark.parametrize("width", [37, 64, 101])
def test_frame_count_matches_backbone(width: int) -> None:
    model = CRNN(num_classes=5, height=32).eval()
    with torch.no_grad():
        log_probs, lengths = model(torch.rand(1, 1, 32, width), torch.tensor([width]))
    assert log_probs.shape[1] == output_length(torch.tensor(width)) == lengths.item()
    assert torch.allclose(log_probs.exp().sum(-1), torch.ones(1, log_probs.shape[1]))


def test_greedy_decode_collapses_repeats_and_blanks() -> None:
    frames = [1, 1, 0, 1, 2, 2, 0, 0]  # a a _ a b b _ _  ->  a a b
    log_probs = torch.full((1, len(frames), 3), -5.0)
    log_probs[0, torch.arange(len(frames)), torch.tensor(frames)] = 0.0
    [(ids, confidence)] = greedy_decode(log_probs, torch.tensor([6]))
    assert ids == [1, 1, 2]
    assert confidence == pytest.approx(1.0)


def test_training_smoke(roots: tuple[Path, Path], tmp_path: Path, monkeypatch) -> None:
    """Two epochs on the fixture pages: checkpoints, history, and evaluation of the result."""
    monkeypatch.setattr("diploma.data.DATA_ROOT", roots[0])
    monkeypatch.setattr("diploma.data.SPLITS_ROOT", roots[1])
    config = TrainConfig(
        name="smoke", train=["makarov:train*2"], val=["makarov:test"], epochs=2, batch_size=2, height=32,
        hidden=32, pretrained=False, workers=0, device="cpu", out=tmp_path,
    )  # fmt: skip
    best = train(config)

    history = [json.loads(l) for l in (tmp_path / "smoke" / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [h["epoch"] for h in history] == [1, 2]
    assert (tmp_path / "smoke" / "last.pt").exists()

    recognizer = CRNNRecognizer.load(best)
    assert recognizer.describe()["epoch"] in (1, 2)
    report = evaluate(recognizer, load_dataset("makarov:test"))
    assert report["lines"] == 1 and "coverage" in report

    # Fine-tuning grows the output layer for unseen symbols and keeps the old ones.
    tuned = train(TrainConfig(name="tuned", init=str(best), train=["peter:train"], val=["makarov:test"], epochs=1,
                              batch_size=2, workers=0, device="cpu", out=tmp_path))  # fmt: skip
    old, new = torch.load(best, weights_only=True), torch.load(tuned, weights_only=True)
    assert new["alphabet"].startswith(old["alphabet"]) and len(new["alphabet"]) > len(old["alphabet"])


def test_cli_train(roots: tuple[Path, Path], tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("diploma.data.DATA_ROOT", roots[0])
    monkeypatch.setattr("diploma.data.SPLITS_ROOT", roots[1])
    main(["train", "--name", "cli", "--train", "makarov:train", "--val", "makarov:test", "--epochs", "1",
          "--batch-size", "2", "--height", "32", "--hidden", "16", "--no-pretrained", "--workers", "0",
          "--device", "cpu", "--out", str(tmp_path)])  # fmt: skip
    config = json.loads((tmp_path / "cli" / "config.json").read_text(encoding="utf-8"))
    assert (config["pretrained"], config["augment"], config["hidden"]) == (False, True, 16)
    assert "best checkpoint" in capsys.readouterr().out


@pytest.mark.parametrize("n", [1, 7, 64, 1000])
def test_width_buckets_cover_every_line_once(n: int) -> None:
    from diploma.training import WidthBuckets

    torch.manual_seed(0)
    widths = torch.rand(n).tolist()
    sampler = WidthBuckets(widths, batch_size=8, pool=4)
    batches = list(sampler)
    assert len(batches) == len(sampler)
    assert sorted(i for b in batches for i in b) == list(range(n))
    assert all(len(b) <= 8 for b in batches)
    # inside a batch, lines come from one sorted pool, so they are ordered by width
    assert all(b == sorted(b, key=widths.__getitem__) for b in batches)


@pytest.mark.parametrize(("capability", "expected"), [((8, 9), torch.bfloat16), ((7, 5), torch.float16)])
def test_mixed_precision_avoids_emulated_bf16(monkeypatch, capability: tuple[int, int], expected: torch.dtype) -> None:
    """RTX 4090 trains in bf16; a T4 can create bf16 tensors but has no bf16 hardware, so it gets fp16."""
    native = capability >= (8, 0)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: native or including_emulation)
    assert mixed_precision(torch.device("cuda"))[0] is expected
