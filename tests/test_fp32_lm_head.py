"""The vocabulary head stays fp32 without taking the rest of the model with it."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from slime.backends.megatron_utils.fp32_lm_head import (
    cast_lm_head_for_rollout,
    promote_vocab_output_layers,
    untie_rollout_embeddings,
    validate_fp32_lm_head_config,
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


class _MatmulHead(_Head):
    def forward(self, input_, weight=None, runtime_gather_output=None):
        return F.linear(input_, self.weight if weight is None else weight), None


@pytest.mark.parametrize("autocast_enabled", [False, True])
def test_head_gemm_gradients_and_optimizer_stay_fp32(autocast_enabled):
    torch.manual_seed(123)
    model = nn.Module()
    model.body = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    model.output_layer = _MatmulHead(16, torch.bfloat16)
    promote_vocab_output_layers(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = model.output_layer.weight.detach().clone()
    context = torch.autocast("cpu", dtype=torch.bfloat16) if autocast_enabled else nullcontext()

    with context:
        hidden = model.body(torch.randn(3, 4, dtype=torch.bfloat16))
        logits, _ = model.output_layer(hidden)
    assert hidden.dtype == torch.bfloat16
    assert logits.dtype == torch.float32
    logits.square().mean().backward()
    assert model.body.weight.grad.dtype == torch.bfloat16
    assert model.output_layer.weight.grad.dtype == torch.float32
    assert torch.isfinite(model.output_layer.weight.grad).all()
    optimizer.step()
    assert model.output_layer.weight.dtype == torch.float32
    assert not torch.equal(before, model.output_layer.weight)
    assert optimizer.state[model.output_layer.weight]["exp_avg"].dtype == torch.float32


def test_deferred_head_gradients_are_rejected_before_dtype_changes():
    model = _Model()
    model.output_layer.config = SimpleNamespace(defer_embedding_wgrad_compute=True)

    with pytest.raises(ValueError, match="does not support --defer-embedding-wgrad-compute"):
        promote_vocab_output_layers(model)
    assert model.output_layer.weight.dtype == torch.bfloat16
    assert model.output_layer.gradient_accumulation_fusion is True


def test_deferred_head_gradients_are_rejected_by_argument_validation(monkeypatch):
    from test_megatron_argument_validation import load_slime_arguments_module, make_slime_validate_args

    args = make_slime_validate_args(fp32_lm_head=True, defer_embedding_wgrad_compute=True)
    module = load_slime_arguments_module(monkeypatch)
    with pytest.raises(ValueError, match="does not support --defer-embedding-wgrad-compute"):
        module.slime_validate_args(args)


def test_immediate_head_gradients_allow_fp32():
    validate_fp32_lm_head_config(SimpleNamespace(defer_embedding_wgrad_compute=False))


def test_megatron_prevalidation_rejects_deferred_head_gradients(monkeypatch):
    from test_megatron_argument_validation import load_arguments_module

    module = load_arguments_module(monkeypatch)
    with pytest.raises(ValueError, match="does not support --defer-embedding-wgrad-compute"):
        module.set_default_megatron_args(SimpleNamespace(fp32_lm_head=True, defer_embedding_wgrad_compute=True))


@pytest.mark.parametrize("base_flag,override_flag", [(False, True), (True, False)])
def test_role_overrides_cannot_change_head_tying(base_flag, override_flag, monkeypatch):
    from test_megatron_argument_validation import load_slime_arguments_module

    module = load_slime_arguments_module(monkeypatch)
    with pytest.raises(ValueError, match="must be set globally"):
        module._apply_megatron_role_overrides(
            SimpleNamespace(fp32_lm_head=base_flag), {"fp32_lm_head": override_flag}, "actor"
        )


@pytest.mark.parametrize("flag", [False, True])
def test_role_overrides_allow_matching_global_flag(flag, monkeypatch):
    from test_megatron_argument_validation import load_slime_arguments_module

    module = load_slime_arguments_module(monkeypatch)
    args = module._apply_megatron_role_overrides(SimpleNamespace(fp32_lm_head=flag), {"fp32_lm_head": flag}, "actor")
    assert args.fp32_lm_head is flag


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
