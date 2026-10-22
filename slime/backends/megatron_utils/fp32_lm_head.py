"""Keep the actor vocabulary head in fp32 while the rest of the model stays bf16.

Megatron's ``Float16Module`` casts every parameter to bf16. The head has to be
promoted again after that cast and before the grad buffers snapshot the dtype.
The vocabulary GEMM then runs in fp32. Rollout engines still receive a bf16 copy.
"""

import json
import logging

import torch

logger = logging.getLogger(__name__)


def validate_fp32_lm_head_config(config) -> None:
    """Native deferred wgrad cannot use this head's non-fused backward path."""
    if getattr(config, "defer_embedding_wgrad_compute", False):
        raise ValueError(
            "--fp32-lm-head does not support --defer-embedding-wgrad-compute; "
            "remove --defer-embedding-wgrad-compute so the FP32 head computes its gradients immediately."
        )


def promote_vocab_output_layers(module: torch.nn.Module) -> int:
    """Cast untied vocabulary heads to fp32 and make their GEMM match that dtype.

    A value head (output width 1) and a tied head (no owned weight) are left alone.
    Returns the number of heads promoted.
    """
    promoted = 0
    for name, layer in module.named_modules():
        if name != "output_layer" and not name.endswith(".output_layer"):
            continue
        weight = getattr(layer, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[0] <= 1:
            continue
        validate_fp32_lm_head_config(getattr(layer, "config", None))
        if weight.dtype != torch.float32:
            weight.data = weight.data.float()
        if hasattr(layer, "gradient_accumulation_fusion"):
            # Use the plain FP32 matmul backward for the vocabulary head. Native
            # deferred wgrad requires fusion and is rejected above.
            layer.gradient_accumulation_fusion = False
        _wrap_fp32_forward(layer)
        promoted += 1
    if promoted:
        logger.info("Keeping %s vocabulary head(s) in fp32.", promoted)
    return promoted


def install_fp32_lm_head_hook() -> None:
    """Promote vocabulary heads after ``Float16Module`` casts the model to bf16."""
    from megatron.core.transformer.module import Float16Module

    if getattr(Float16Module, "_slime_fp32_lm_head_installed", False):
        return
    original_init = Float16Module.__init__

    def _init(self, config, module):
        original_init(self, config, module)
        promote_vocab_output_layers(self.module)

    Float16Module.__init__ = _init
    Float16Module._slime_fp32_lm_head_installed = True


def cast_lm_head_for_rollout(args, named_tensors):
    """Send the rollout server a bf16 LM head. Training keeps the fp32 parameter."""
    if not getattr(args, "fp32_lm_head", False):
        return named_tensors
    cast = []
    for name, tensor in named_tensors:
        if name.endswith("lm_head.weight") and tensor.dtype != torch.bfloat16:
            tensor = tensor.detach().to(dtype=torch.bfloat16)
        cast.append((name, tensor))
    return cast


def untie_rollout_embeddings(overrides: str, *, has_text_config: bool) -> str:
    """Give SGLang its own LM head so a synced head does not overwrite the embeddings.

    Qwen3.5-4B ties ``lm_head`` to the token embedding. After this training flag
    unties them, the server has to allocate a separate head or the two updates
    land on one parameter.
    """
    values = json.loads(overrides or "{}")
    values["tie_word_embeddings"] = False
    if has_text_config:
        text = dict(values.get("text_config") or {})
        text["tie_word_embeddings"] = False
        values["text_config"] = text
    return json.dumps(values)


def _wrap_fp32_forward(layer: torch.nn.Module) -> None:
    if getattr(layer, "_slime_fp32_lm_head", False):
        return
    original = layer.forward

    def forward(input_, weight=None, runtime_gather_output=None):
        # Casting operands alone is insufficient: an enclosing AMP context can
        # cast the GEMM straight back to bf16. Preserve FP32 computation as well
        # as FP32 parameter storage.
        with torch.autocast(device_type=input_.device.type, enabled=False):
            if input_.is_floating_point() and input_.dtype != torch.float32:
                input_ = input_.float()
            if weight is not None and weight.is_floating_point() and weight.dtype != torch.float32:
                weight = weight.float()
            return original(input_, weight, runtime_gather_output)

    layer.forward = forward
    layer._slime_fp32_lm_head = True
