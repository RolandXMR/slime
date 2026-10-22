"""The vocabulary head stays fp32 without taking the rest of the model with it."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from slime.backends.megatron_utils.fp32_lm_head import (
    cast_lm_head_for_rollout,
    fp32_lm_head_hook,
    promote_vocab_output_layers,
    untie_rollout_embeddings,
)


NUM_GPUS = 0


class _Head(nn.Module):
    def __init__(self, out_features, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(out_features, 4, dtype=dtype))
        self.gradient_accumulation_fusion = True
        self.seen = None

    def forward(self, input_, weight=None, runtime_gather_output=None):
        self.seen = input_.dtype
        return F.linear(input_, self.weight if weight is None else weight), None


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

    assert promote_vocab_output_layers(model) == 1
    model.output_layer(torch.zeros(2, 4, dtype=torch.bfloat16))
    assert model.output_layer.seen == torch.float32


@pytest.mark.parametrize("autocast_enabled", [False, True])
def test_head_gemm_gradients_and_optimizer_stay_fp32(autocast_enabled):
    torch.manual_seed(123)
    model = nn.Module()
    model.body = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    model.output_layer = _Head(16, torch.bfloat16)
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


@pytest.fixture
def native_model(tmp_path):
    layers = pytest.importorskip("megatron.core.tensor_parallel.layers")
    module = pytest.importorskip("megatron.core.transformer.module")
    config_module = pytest.importorskip("megatron.core.transformer.transformer_config")
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Native CPU smoke requires Gloo")

    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)

    class Model(torch.nn.Module):
        def __init__(self, *, deferred=False):
            super().__init__()
            self.config = config_module.TransformerConfig(
                num_layers=2,
                hidden_size=4,
                num_attention_heads=1,
                ffn_hidden_size=8,
                params_dtype=torch.bfloat16,
                bf16=True,
                use_cpu_initialization=True,
                perform_initialization=False,
                gradient_accumulation_fusion=True,
                pipeline_model_parallel_size=2 if deferred else 1,
                pipeline_dtype=torch.bfloat16,
                defer_embedding_wgrad_compute=deferred,
            )
            self.pg_collection = SimpleNamespace(pp=dist.group.WORLD)
            self.body = torch.nn.Linear(4, 4, bias=False)
            self.output_layer = layers.ColumnParallelLinear(
                4,
                8,
                config=self.config,
                init_method=lambda _: None,
                bias=False,
                gather_output=False,
                tp_group=dist.group.WORLD,
                embedding_activation_buffer=[],
                grad_output_buffer=[],
            )
            with torch.no_grad():
                self.output_layer.weight.uniform_(-0.1, 0.1)

        def forward(self, x):
            return self.output_layer(self.body(x))[0]

    try:
        yield Model, module.Float16Module
    finally:
        dist.destroy_process_group()


def test_native_head_stays_fp32_under_autocast_backward_and_optimizer(native_model):
    Model, Float16Module = native_model
    model = Model()
    original_init = Float16Module.__init__
    with fp32_lm_head_hook():
        wrapped = Float16Module(model.config, model)
    assert Float16Module.__init__ is original_init
    ordinary = Model()
    Float16Module(ordinary.config, ordinary)
    assert ordinary.output_layer.weight.dtype == torch.bfloat16
    head_dtypes = []
    handle = model.output_layer.register_forward_hook(lambda _layer, _inputs, out: head_dtypes.append(out[0].dtype))
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = wrapped(torch.randn(2, 1, 4))
    finally:
        handle.remove()
    assert head_dtypes == [torch.float32]
    assert model.body.weight.dtype == torch.bfloat16
    assert model.output_layer.weight.dtype == torch.float32
    logits.square().mean().backward()
    assert model.output_layer.weight.grad.dtype == torch.float32
    assert model.body.weight.grad.dtype == torch.bfloat16
    before = model.output_layer.weight.detach().clone()
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=0.01)
    optimizer.step()
    assert not torch.equal(before, model.output_layer.weight)
    assert optimizer.state[model.output_layer.weight]["exp_avg"].dtype == torch.float32

    state = model.state_dict()
    state["output_layer.weight"] = state["output_layer.weight"].bfloat16()
    model.load_state_dict(state)
    assert model.output_layer.weight.dtype == torch.float32


def test_native_deferred_head_is_rejected_before_backward(native_model):
    Model, Float16Module = native_model
    model = Model(deferred=True)
    original_init = Float16Module.__init__
    with pytest.raises(ValueError, match="does not support --defer-embedding-wgrad-compute"), fp32_lm_head_hook():
        Float16Module(model.config, model)
    assert Float16Module.__init__ is original_init
