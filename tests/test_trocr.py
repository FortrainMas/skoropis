import pytest
import torch
from transformers import RobertaConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

from diploma.models.trocr import TrOCR

PAD, BOS, EOS = 0, 1, 2


@pytest.fixture(scope="module")
def tiny() -> TrOCR:
    """A randomly initialized two-layer encoder-decoder: no download, same code paths."""
    encoder = ViTConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64, image_size=32, patch_size=16
    )
    decoder = RobertaConfig(
        vocab_size=50, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
        is_decoder=True, add_cross_attention=True, pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS,
    )  # fmt: skip
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder, decoder)
    torch.manual_seed(0)
    model = VisionEncoderDecoderModel(config).eval()
    for cfg in (model.config, model.generation_config):
        cfg.decoder_start_token_id, cfg.pad_token_id, cfg.eos_token_id = BOS, PAD, EOS
    recognizer = TrOCR.__new__(TrOCR)  # skip from_pretrained
    recognizer.model = model
    return recognizer


@pytest.mark.parametrize("generation", [{"num_beams": 1, "do_sample": False}, {"num_beams": 3}])
@torch.inference_mode()
def test_confidence_is_geometric_mean_token_probability(tiny: TrOCR, generation: dict) -> None:
    pixels = torch.randn(3, 3, 32, 32)
    out = tiny.model.generate(pixels, max_length=12, output_scores=True, return_dict_in_generate=True, **generation)

    # Rescore the chosen sequences with a plain forward pass.
    tokens = out.sequences[:, 1:]
    log_probs = tiny.model(pixel_values=pixels, decoder_input_ids=out.sequences[:, :-1]).logits.log_softmax(-1)
    chosen = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    mask = tokens != PAD
    expected = (chosen.masked_fill(~mask, 0).sum(1) / mask.sum(1)).exp()

    assert tiny._confidences(out) == pytest.approx(expected.tolist(), rel=1e-4)
