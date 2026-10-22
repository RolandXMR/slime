"""Load tied Megatron checkpoints into an actor with a separate FP32 head."""

import dataclasses
import logging
from contextlib import contextmanager
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _is_vocab_head(key, tensor):
    return (
        not key.startswith("optimizer.")
        and key.endswith("output_layer.weight")
        and len(tensor.global_shape) == 2
        and tensor.global_shape[0] > 1
    )


def _prepare_fp32_head_load(state_dict, metadata, checkpoint_args, *, optimizer_requested=False):
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    tied = getattr(checkpoint_args, "untie_embeddings_and_output_weights", None) is False
    # This decision must agree on every PP rank before any collective load. A
    # changed dtype or an added head changes distributed-optimizer buffer layout.
    changed_layout = tied or any(
        _is_vocab_head(key, tensor) and tensor.dtype != torch.float32 for key, tensor in metadata.items()
    )
    if (optimizer_requested or "optimizer" in state_dict) and changed_layout:
        raise ValueError(
            "Cannot restore optimizer state while --fp32-lm-head changes tied embeddings or the head dtype. Use --no-load-optim or --finetune to initialize a fresh optimizer; checkpoints saved with a separate FP32 head can resume normally."
        )

    def retarget(value):
        if isinstance(value, dict):
            return {key: retarget(item) for key, item in value.items()}
        if isinstance(value, list):
            return [retarget(item) for item in value]
        if not isinstance(value, ShardedTensor):
            return value
        if tied and value.key.endswith("embedding.word_embeddings.weight") and value.key in metadata:
            source_dtype = metadata[value.key].dtype
            if value.dtype != source_dtype:
                # All replicas of an aliased key must request the same dtype.
                # Keep full stored precision for the head, then let model load
                # copy the embedding back to its actual BF16 parameter.
                data = torch.empty_like(value.data, dtype=source_dtype) if value.data is not None else None
                return dataclasses.replace(value, dtype=source_dtype, data=data)
        if not _is_vocab_head(value.key, value):
            return value
        if value.key in metadata or not tied:
            return value
        source_key = value.key.removesuffix("output_layer.weight") + "embedding.word_embeddings.weight"
        if source_key not in metadata:
            return value

        source = metadata[source_key]
        # The embedding is still a primary shard elsewhere in the model. Match
        # its stored dtype and mark this second request as a replica, exactly as
        # Megatron does for a tied head on the last pipeline stage. Mixed dtypes
        # for one sharded key otherwise fail metadata validation. load_state_dict
        # later copies this stored-dtype shard into the existing FP32 parameter.
        data = torch.empty_like(value.data, dtype=source.dtype) if value.data is not None else None
        replica_id = 1 if isinstance(value.replica_id, int) else (1, *value.replica_id[1:])
        logger.info("Initializing separate FP32 head %s from tied checkpoint tensor %s.", value.key, source_key)
        return dataclasses.replace(value, key=source_key, dtype=source.dtype, data=data, replica_id=replica_id)

    return {
        key: retarget(value) if key == "model" or (key.startswith("model") and key[5:].isdigit()) else value
        for key, value in state_dict.items()
    }


@contextmanager
def fp32_head_checkpoint_loading(*, optimizer_requested=False):
    """Retarget model-only loads without rewriting checkpoints or model parameters."""
    from megatron.core import dist_checkpointing

    original_load = dist_checkpointing.load

    def load(state_dict, checkpoint_dir, *args, **kwargs):
        metadata = dist_checkpointing.load_tensors_metadata(checkpoint_dir)
        common = dist_checkpointing.load_common_state_dict(checkpoint_dir)
        checkpoint_args = common.get("args")
        load_optimizer = (
            optimizer_requested
            and Path(checkpoint_dir).name != "release"
            and not getattr(checkpoint_args, "no_save_optim", False)
        )
        state_dict = _prepare_fp32_head_load(state_dict, metadata, checkpoint_args, optimizer_requested=load_optimizer)
        return original_load(state_dict, checkpoint_dir, *args, **kwargs)

    dist_checkpointing.load = load
    try:
        yield
    finally:
        dist_checkpointing.load = original_load
