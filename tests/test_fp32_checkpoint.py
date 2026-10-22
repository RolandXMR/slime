"""Real torch_dist loads for a tied BF16 checkpoint and separate FP32 head."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core.dist_checkpointing")
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.core import CheckpointingConfig, save_config
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelLoadStrategyWrapper
from megatron.core.dist_checkpointing.strategies.torch import TorchDistLoadShardedStrategy

from slime.backends.megatron_utils import checkpoint as _checkpoint_adapter  # noqa: F401
from slime.backends.megatron_utils.fp32_checkpoint import _prepare_fp32_head_load, fp32_head_checkpoint_loading

NUM_GPUS = 0


@pytest.fixture
def process_group(tmp_path):
    if torch.distributed.is_initialized():
        yield torch.distributed.group.WORLD
        return
    torch.distributed.init_process_group("gloo", init_method=f"file://{tmp_path}/gloo", rank=0, world_size=1)
    try:
        yield torch.distributed.group.WORLD
    finally:
        torch.distributed.destroy_process_group()


def _shard(key, data, replica_id=0):
    return ShardedTensor.from_rank_offsets(key, data, replica_id=replica_id)


def _save_cpu_checkpoint(state, path):
    # Synchronous PyTorch DCP writes MCore's tensor storage without its GPU-only
    # async writer. common.pt and the descriptor match older tied references.
    tensors = {shard.key: shard.data for shard in state["model"].values()}
    torch.distributed.checkpoint.save(tensors, checkpoint_id=str(path))
    if torch.distributed.get_rank() == 0:
        torch.save({"args": state["args"]}, path / "common.pt")
        save_config(CheckpointingConfig("torch_dist", 1), str(path))
    torch.distributed.barrier()


def _save(path, *, tied=True, head_dtype=torch.bfloat16, embedding_dtype=torch.bfloat16):
    path.mkdir()
    embedding = (torch.arange(24, dtype=torch.float32).reshape(6, 4) / 8 + 0.001).to(embedding_dtype)
    model = {"embedding.word_embeddings.weight": _shard("embedding.word_embeddings.weight", embedding)}
    if not tied:
        head = embedding.float() + 1.001 if head_dtype == torch.float32 else (embedding + 1).to(head_dtype)
        model["output_layer.weight"] = _shard("output_layer.weight", head)
    _save_cpu_checkpoint(
        {"args": SimpleNamespace(untie_embeddings_and_output_weights=not tied), "model": model},
        path,
    )
    return embedding


@pytest.mark.parametrize("fully_parallel", [False, True])
@pytest.mark.parametrize("source_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_tied_checkpoint_initializes_separate_fp32_head(tmp_path, process_group, fully_parallel, source_dtype):
    path = tmp_path / "checkpoint"
    expected = _save(path, embedding_dtype=source_dtype)
    model = torch.nn.Module()
    model.embedding = torch.nn.Module()
    model.embedding.word_embeddings = torch.nn.Embedding(6, 4, dtype=torch.bfloat16)
    model.output_layer = torch.nn.Linear(4, 6, bias=False, dtype=torch.float32)
    embedding = model.embedding.word_embeddings.weight
    head = model.output_layer.weight
    requested = {
        "model": {
            "embedding.word_embeddings.weight": _shard("embedding.word_embeddings.weight", embedding),
            "output_layer.weight": _shard("output_layer.weight", head),
        }
    }
    original_key = requested["model"]["output_layer.weight"].key
    strategy = TorchDistLoadShardedStrategy()
    if fully_parallel:
        strategy = FullyParallelLoadStrategyWrapper(strategy, process_group)
    before = {file.name: file.read_bytes() for file in path.iterdir() if file.is_file()}
    original_load = dist_checkpointing.load
    with fp32_head_checkpoint_loading():
        loaded = dist_checkpointing.load(requested, str(path), strategy, validate_access_integrity=True)
    assert dist_checkpointing.load is original_load
    assert requested["model"]["output_layer.weight"].key == original_key
    model.load_state_dict(loaded["model"])
    assert head.dtype == torch.float32 and head.data_ptr() != embedding.data_ptr()
    torch.testing.assert_close(head, expected.float(), rtol=0, atol=0)
    torch.testing.assert_close(embedding, expected.bfloat16(), rtol=0, atol=0)
    assert before == {file.name: file.read_bytes() for file in path.iterdir() if file.is_file()}


@pytest.mark.parametrize("head_dtype", [torch.bfloat16, torch.float32])
def test_existing_untied_head_loads_without_embedding_alias(tmp_path, process_group, head_dtype):
    path = tmp_path / "checkpoint"
    embedding = _save(path, tied=False, head_dtype=head_dtype)
    target = torch.empty_like(embedding, dtype=torch.float32)
    requested = {"model": {"output_layer.weight": _shard("output_layer.weight", target)}}
    with fp32_head_checkpoint_loading():
        loaded = dist_checkpointing.load(requested, str(path), validate_access_integrity=False)
    expected = embedding.float() + 1.001 if head_dtype == torch.float32 else (embedding + 1).float()
    torch.testing.assert_close(loaded["model"]["output_layer.weight"], expected, rtol=0, atol=0)


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("owns_optimizer", [False, True])
def test_changed_layout_rejects_optimizer_resume_on_every_pipeline_rank(tied, owns_optimizer):
    metadata = {
        "embedding.word_embeddings.weight": _shard(
            "embedding.word_embeddings.weight", torch.zeros(6, 4, dtype=torch.bfloat16)
        )
    }
    if not tied:
        metadata["output_layer.weight"] = _shard("output_layer.weight", torch.zeros(6, 4, dtype=torch.bfloat16))
    # A first PP rank owns no head. It must still reject before other ranks load.
    state = {"model": {}}
    if owns_optimizer:
        state["optimizer"] = {}
    with pytest.raises(ValueError, match="--no-load-optim"):
        _prepare_fp32_head_load(
            state, metadata, SimpleNamespace(untie_embeddings_and_output_weights=not tied), optimizer_requested=True
        )


def test_fp32_untied_checkpoint_keeps_optimizer_resume():
    metadata = {
        "output_layer.weight": _shard("output_layer.weight", torch.zeros(6, 4, dtype=torch.float32)),
        "optimizer.state.exp_avg.output_layer.weight": _shard(
            "optimizer.state.exp_avg.output_layer.weight", torch.zeros(6, 4, dtype=torch.bfloat16)
        ),
    }
    state = {"model": {}, "optimizer": {"state": "unchanged"}}
    assert _prepare_fp32_head_load(state, metadata, SimpleNamespace(untie_embeddings_and_output_weights=True)) == state


def test_missing_untied_head_is_not_repaired_as_tied():
    head = _shard("output_layer.weight", torch.zeros(6, 4, dtype=torch.float32))
    metadata = {
        "embedding.word_embeddings.weight": _shard(
            "embedding.word_embeddings.weight", torch.zeros(6, 4, dtype=torch.bfloat16)
        )
    }
    state = {"model": {"output_layer.weight": head}}
    prepared = _prepare_fp32_head_load(state, metadata, SimpleNamespace(untie_embeddings_and_output_weights=True))
    assert prepared["model"]["output_layer.weight"] is head


def test_load_hook_is_restored_after_loader_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("checkpoint read failed")

    monkeypatch.setattr(dist_checkpointing, "load", fail)
    monkeypatch.setattr(dist_checkpointing, "load_tensors_metadata", lambda path: {})
    monkeypatch.setattr(dist_checkpointing, "load_common_state_dict", lambda path: {})
    with pytest.raises(RuntimeError, match="checkpoint read failed"):
        with fp32_head_checkpoint_loading():
            dist_checkpointing.load({"model": {}}, "checkpoint")
    assert dist_checkpointing.load is fail


@pytest.mark.parametrize("checkpoint_name,no_save_optim", [("release", False), ("iter_0000007", True)])
def test_model_only_checkpoint_does_not_trigger_optimizer_guard(monkeypatch, checkpoint_name, no_save_optim):
    metadata = {
        "embedding.word_embeddings.weight": _shard(
            "embedding.word_embeddings.weight", torch.zeros(6, 4, dtype=torch.bfloat16)
        )
    }
    common = {"args": SimpleNamespace(untie_embeddings_and_output_weights=False, no_save_optim=no_save_optim)}
    monkeypatch.setattr(dist_checkpointing, "load_tensors_metadata", lambda path: metadata)
    monkeypatch.setattr(dist_checkpointing, "load_common_state_dict", lambda path: common)
    monkeypatch.setattr(dist_checkpointing, "load", lambda state, *args, **kwargs: state)
    with fp32_head_checkpoint_loading(optimizer_requested=True):
        assert dist_checkpointing.load({"model": {}}, checkpoint_name) == {"model": {}}


def _distributed_tied_load_worker(rank, layout, init_file, path):
    torch.distributed.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        full = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4) / 8
        expected = full.chunk(2)[rank].clone() if layout == "tp" else full
        offsets = ((0, rank, 2),) if layout == "tp" else ()
        saved_model = {"embedding.word_embeddings.weight": _shard("embedding.word_embeddings.weight", full)}
        _save_cpu_checkpoint(
            {"args": SimpleNamespace(untie_embeddings_and_output_weights=False), "model": saved_model},
            Path(path),
        )
        requested = {"model": {}}
        if layout == "tp" or rank == 0:
            requested["model"]["embedding.word_embeddings.weight"] = ShardedTensor.from_rank_offsets(
                "embedding.word_embeddings.weight", torch.empty_like(expected), *offsets
            )
        head = None
        if layout == "tp" or rank == 1:
            head = torch.nn.Parameter(torch.empty_like(expected, dtype=torch.float32))
            requested["model"]["output_layer.weight"] = ShardedTensor.from_rank_offsets(
                "output_layer.weight", head, *offsets
            )
        # TP/PP ranks each have a singleton DP/CP group when DP=CP=1.
        groups = [torch.distributed.new_group([member], backend="gloo") for member in range(2)]
        strategy = FullyParallelLoadStrategyWrapper(TorchDistLoadShardedStrategy(), groups[rank])
        with fp32_head_checkpoint_loading():
            loaded = dist_checkpointing.load(requested, path, strategy, validate_access_integrity=True)
        if head is not None:
            with torch.no_grad():
                head.copy_(loaded["model"]["output_layer.weight"])
            assert head.dtype == torch.float32
            torch.testing.assert_close(head, expected.float(), rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("layout", ["tp", "pp"])
def test_tied_checkpoint_loads_across_two_ranks(tmp_path, layout):
    path = tmp_path / "checkpoint"
    path.mkdir()
    torch.multiprocessing.spawn(
        _distributed_tied_load_worker,
        args=(layout, str(tmp_path / "gloo"), str(path)),
        nprocs=2,
        join=True,
    )
