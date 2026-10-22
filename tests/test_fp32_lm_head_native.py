"""CPU checks against native Megatron when its runtime is installed."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from slime.backends.megatron_utils.fp32_lm_head import install_fp32_lm_head_hook

NUM_GPUS = 0


@pytest.fixture
def native_model(tmp_path):
    layers = pytest.importorskip("megatron.core.tensor_parallel.layers")
    module = pytest.importorskip("megatron.core.transformer.module")
    config_module = pytest.importorskip("megatron.core.transformer.transformer_config")
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Native CPU smoke requires Gloo")

    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    original_init = module.Float16Module.__init__
    installed = getattr(module.Float16Module, "_slime_fp32_lm_head_installed", False)
    install_fp32_lm_head_hook()

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
        module.Float16Module.__init__ = original_init
        if not installed:
            del module.Float16Module._slime_fp32_lm_head_installed
        dist.destroy_process_group()


def test_native_head_stays_fp32_under_autocast_backward_and_optimizer(native_model):
    Model, Float16Module = native_model
    model = Model()
    wrapped = Float16Module(model.config, model)
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
    with pytest.raises(ValueError, match="does not support --defer-embedding-wgrad-compute"):
        Float16Module(model.config, model)
