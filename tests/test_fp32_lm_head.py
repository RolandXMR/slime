"""The vocabulary head stays fp32 without taking the rest of the model with it."""

import json

import torch
from torch import nn

from slime.backends.megatron_utils.fp32_lm_head import (
    cast_lm_head_for_rollout,
    promote_vocab_output_layers,
    untie_rollout_embeddings,
)


class _Head(nn.Module):
    def __init__(self, out_features, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(out_features, 4, dtype=dtype))
        self.gradient_accumulation_fusion = True
        self.seen = None

    def forward(self, input_, weight=None, runtime_gather_output=None):
        self.seen = input_.dtype
        return input_


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_layer = _Head(16, torch.bfloat16)
        self.value_head = nn.Module()
        self.language_model = nn.Module()
        self.language_model.output_layer = _Head(1, torch.bfloat16)


def test_promote_casts_vocab_head_and_upcasts_the_gemm_input():
    model = _Model()
    assert promote_vocab_output_layers(model) == 1
    assert model.output_layer.weight.dtype == torch.float32
    assert model.output_layer.gradient_accumulation_fusion is False
    assert model.language_model.output_layer.weight.dtype == torch.bfloat16

    model.output_layer(torch.zeros(2, 4, dtype=torch.bfloat16))
    assert model.output_layer.seen == torch.float32


def test_promote_is_idempotent():
    model = _Model()
    promote_vocab_output_layers(model)
    promote_vocab_output_layers(model)
    model.output_layer(torch.zeros(2, 4, dtype=torch.float32))
    assert model.output_layer.seen == torch.float32


def test_rollout_copy_of_the_head_is_bf16():
    class Args:
        fp32_lm_head = True

    head = torch.ones(3, 4, dtype=torch.float32)
    body = torch.ones(3, 4, dtype=torch.bfloat16)
    converted = cast_lm_head_for_rollout(
        Args(),
        [("lm_head.weight", head), ("model.embed_tokens.weight", body)],
    )
    assert converted[0][1].dtype == torch.bfloat16
    assert converted[1][1].dtype == torch.bfloat16
    assert head.dtype == torch.float32


def test_rollout_server_unties_a_tied_checkpoint():
    overrides = untie_rollout_embeddings(json.dumps({"dtype": "bfloat16"}), has_text_config=True)
    values = json.loads(overrides)
    assert values["tie_word_embeddings"] is False
    assert values["text_config"]["tie_word_embeddings"] is False
    assert values["dtype"] == "bfloat16"
