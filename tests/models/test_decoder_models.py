import torch

from semantic_cloud.models.cfrm_decoder import CFRMDecoder
from semantic_cloud.models.gru_decoder import GRUDecoder
from semantic_cloud.models.transformer_decoder import TinyTransformerDecoder


def test_transformer_decoder_returns_vocab_logits():
    model = TinyTransformerDecoder(vocab_size=128)
    tokens = torch.randint(0, 128, (4, 20))

    logits = model(tokens)

    assert logits.shape == (4, 20, 128)


def test_gru_decoder_returns_vocab_logits():
    model = GRUDecoder(vocab_size=128)
    tokens = torch.randint(0, 128, (4, 20))

    logits = model(tokens)

    assert logits.shape == (4, 20, 128)


def test_cfrm_decoder_returns_vocab_logits():
    model = CFRMDecoder(vocab_size=128, num_clouds=4, hidden_dim=64)
    tokens = torch.randint(0, 128, (4, 20))

    logits = model(tokens)

    assert logits.shape == (4, 20, 128)


def test_cfrm_decoder_can_return_state():
    model = CFRMDecoder(vocab_size=128, num_clouds=4, hidden_dim=64)
    tokens = torch.randint(0, 128, (2, 12))

    output = model(tokens, return_state=True)

    assert output["logits"].shape == (2, 12, 128)
    assert "core" in output
