import importlib.util
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from slime.backends.megatron_utils.hf_checkpoint_saver import (
    _clear_existing_hf_weights,
    _copy_hf_assets,
    _finalize_local_shards,
    _SafetensorShardWriter,
    _write_pending_chunk,
    save_hf_model_to_path,
)

from slime.backends.megatron_utils.hf_to_megatron.common import load_model_hf_weights
from slime.backends.megatron_utils.hf_to_megatron.qwen3_5 import qwen3_5_hf_tensor

NUM_GPUS = 0


def test_copy_hf_assets_keeps_quantized_config_and_skips_weights(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    config = {"model_type": "tiny", "quantization_config": {"quant_method": "fp8"}}
    (src / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (src / "tokenizer.json").write_text("{}", encoding="utf-8")
    (src / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (src / "model-00001-of-00001.safetensors").write_bytes(b"weight")
    (src / "pytorch_model.bin").write_bytes(b"weight")

    _copy_hf_assets(str(src), dst)

    assert json.loads((dst / "config.json").read_text(encoding="utf-8")) == config
    assert (dst / "tokenizer.json").exists()
    assert not (dst / "model.safetensors.index.json").exists()
    assert not (dst / "model-00001-of-00001.safetensors").exists()
    assert not (dst / "pytorch_model.bin").exists()


def test_clear_existing_hf_weights_removes_old_weight_files_only(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"weight")
    (tmp_path / "pytorch_model.bin").write_bytes(b"weight")

    _clear_existing_hf_weights(tmp_path)

    assert (tmp_path / "config.json").exists()
    assert not (tmp_path / "model.safetensors.index.json").exists()
    assert not (tmp_path / "model-00001-of-00001.safetensors").exists()
    assert not (tmp_path / "pytorch_model.bin").exists()


def test_save_hf_model_to_path_rejects_origin_checkpoint(tmp_path: Path):
    args = SimpleNamespace(hf_checkpoint=str(tmp_path))

    with pytest.raises(ValueError, match="same directory as --hf-checkpoint"):
        save_hf_model_to_path(args, tmp_path, model=None)


def test_safetensor_shard_writer_writes_hf_index(tmp_path: Path):
    writer = _SafetensorShardWriter(tmp_path, enabled=True)
    writer.write([("layers.0.weight", torch.ones(2, 2)), ("layers.0.weight_scale", torch.ones(1))], shard_idx=0)
    writer.write([("layers.1.weight", torch.zeros(2, 2))], shard_idx=1)
    writer.finalize()

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["metadata"]["total_size"] == 36
    assert index["weight_map"] == {
        "layers.0.weight": "model-00001-of-00002.safetensors",
        "layers.0.weight_scale": "model-00001-of-00002.safetensors",
        "layers.1.weight": "model-00002-of-00002.safetensors",
    }

    shard0 = load_file(tmp_path / "model-00001-of-00002.safetensors")
    shard1 = load_file(tmp_path / "model-00002-of-00002.safetensors")
    assert torch.equal(shard0["layers.0.weight"], torch.ones(2, 2))
    assert torch.equal(shard1["layers.1.weight"], torch.zeros(2, 2))


def test_finalize_shard_files_merges_node_writer_states(tmp_path: Path):
    writer0 = _SafetensorShardWriter(tmp_path, enabled=True)
    writer1 = _SafetensorShardWriter(tmp_path, enabled=True)

    writer0.write([("layers.0.weight", torch.ones(2, 2))], shard_idx=0)
    writer1.write([("layers.1.weight", torch.zeros(2, 2))], shard_idx=1)

    # each rank renames its own files off the shared plan; rank 0 writes the index
    states = [writer0.state(), writer1.state()]
    for rank, state in enumerate(states):
        _finalize_local_shards(tmp_path, state, states, write_index=rank == 0)

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["metadata"]["total_size"] == 32
    assert index["weight_map"] == {
        "layers.0.weight": "model-00001-of-00002.safetensors",
        "layers.1.weight": "model-00002-of-00002.safetensors",
    }
    assert not (tmp_path / "model-00001.safetensors").exists()
    assert not (tmp_path / "model-00002.safetensors").exists()

    shard0 = load_file(tmp_path / "model-00001-of-00002.safetensors")
    shard1 = load_file(tmp_path / "model-00002-of-00002.safetensors")
    assert torch.equal(shard0["layers.0.weight"], torch.ones(2, 2))
    assert torch.equal(shard1["layers.1.weight"], torch.zeros(2, 2))


def test_pending_chunk_write_flushes_incomplete_node_group(tmp_path: Path):
    num_nodes = 3
    writers = [_SafetensorShardWriter(tmp_path, enabled=True) for _ in range(num_nodes)]
    pending_writes = [None] * num_nodes

    for chunk_idx in range(5):
        node_rank = chunk_idx % num_nodes
        pending_writes[node_rank] = (
            chunk_idx,
            [(f"layers.{chunk_idx}.weight", torch.full((1,), chunk_idx, dtype=torch.float32))],
        )

        if (chunk_idx + 1) % num_nodes == 0:
            for i, writer in enumerate(writers):
                pending_writes[i] = _write_pending_chunk(writer, pending_writes[i])

    for i, writer in enumerate(writers):
        pending_writes[i] = _write_pending_chunk(writer, pending_writes[i])

    states = [writer.state() for writer in writers]
    for rank, state in enumerate(states):
        _finalize_local_shards(tmp_path, state, states, write_index=rank == 0)

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["weight_map"] == {f"layers.{i}.weight": f"model-{i + 1:05d}-of-00005.safetensors" for i in range(5)}


def _fp32_args(source):
    return SimpleNamespace(
        hf_checkpoint=str(source),
        fp32_lm_head=True,
        vocab_size=4,
        padded_vocab_size=4,
        q_lora_rank=None,
    )


@pytest.mark.parametrize("script_name", ["convert_torch_dist_to_hf.py", "convert_torch_dist_to_hf_parallel.py"])
def test_checkpoint_conversion_tools_preserve_fp32_head_and_untied_config(tmp_path, script_name):
    path = Path(__file__).resolve().parents[2] / "tools" / script_name
    spec = importlib.util.spec_from_file_location("fp32_hf_conversion_test", path)
    converter = importlib.util.module_from_spec(spec)
    original_unpickler = pickle.Unpickler
    try:
        spec.loader.exec_module(converter)
    finally:
        # These CLI tools install a checkpoint unpickler at module import.
        pickle.Unpickler = original_unpickler

    source, output = tmp_path / "source", tmp_path / "converted"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"tie_word_embeddings": True, "text_config": {"tie_word_embeddings": True}}),
        encoding="utf-8",
    )
    (source / "pytorch_model.bin").write_bytes(b"stale pretrained weights")
    (source / "pytorch_model.bin.index.json").write_text("{}", encoding="utf-8")
    state = {
        "embedding.word_embeddings.weight": torch.full((4, 3), 2.0, dtype=torch.bfloat16),
        "output_layer.weight": torch.full((4, 3), 1.001, dtype=torch.float32),
    }
    converter.save_tensors(_fp32_args(source), "qwen3_5", state, str(output), chunk_size=1024)
    converter._copy_hf_assets(str(source), output, untie_embeddings=True)
    assert not (output / "pytorch_model.bin").exists()
    assert not (output / "pytorch_model.bin.index.json").exists()

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["tie_word_embeddings"] is False
    assert config["text_config"]["tie_word_embeddings"] is False
    index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))
    head = load_file(output / index["weight_map"]["lm_head.weight"])["lm_head.weight"]
    assert head.dtype == torch.float32
    assert torch.equal(head, state["output_layer.weight"])
    assert not torch.equal(head, state["output_layer.weight"].to(torch.bfloat16).float())


@pytest.mark.parametrize("nested_text_config", [False, True])
@pytest.mark.parametrize("for_rollout", [False, True])
def test_saved_head_round_trip_preserves_untying_and_requested_precision(
    tmp_path, monkeypatch, nested_text_config, for_rollout
):
    from slime.backends.megatron_utils.update_weight import common
    from slime.backends.megatron_utils.update_weight import hf_weight_iterator_direct as iterator_module

    source = tmp_path / "source"
    output = tmp_path / "saved"
    source.mkdir()
    config = {"model_type": "qwen3_5", "tie_word_embeddings": True, "marker": "keep"}
    if nested_text_config:
        config["text_config"] = {"tie_word_embeddings": True, "hidden_size": 3}
    original_config = json.dumps(config)
    (source / "config.json").write_text(original_config, encoding="utf-8")

    embedding_name = "module.module.embedding.word_embeddings.weight"
    head_name = "module.module.output_layer.weight"
    source_weights = {
        embedding_name: torch.nn.Parameter(torch.full((4, 3), 2.0, dtype=torch.bfloat16)),
        head_name: torch.nn.Parameter(torch.full((4, 3), 1.001, dtype=torch.float32)),
    }
    model = [SimpleNamespace(weights=source_weights)]
    args = _fp32_args(source)

    # Only distributed collection is replaced. Conversion, config persistence,
    # safetensors serialization, shard indexing and HF loading run unchanged.
    monkeypatch.setattr(common, "named_params_and_buffers", lambda _args, chunks: iter(chunks[0].weights.items()))
    monkeypatch.setattr(
        iterator_module,
        "_get_megatron_local_param_info_buckets",
        lambda _args, _model: [[SimpleNamespace(name=name) for name in source_weights]],
    )
    monkeypatch.setattr(
        iterator_module,
        "_get_megatron_full_params",
        lambda infos, weights: [weights[info.name] for info in infos],
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    save_hf_model_to_path(args, output, model, model_name="qwen3_5", for_rollout=for_rollout)

    saved_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert saved_config["tie_word_embeddings"] is False
    if nested_text_config:
        assert saved_config["text_config"]["tie_word_embeddings"] is False
        assert saved_config["text_config"]["hidden_size"] == 3
    assert saved_config["marker"] == "keep"
    assert (source / "config.json").read_text(encoding="utf-8") == original_config

    index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))
    tensors = load_file(output / index["weight_map"]["lm_head.weight"])
    saved_head = tensors["lm_head.weight"]
    expected_head = source_weights[head_name].detach()
    if for_rollout:
        expected_head = expected_head.to(torch.bfloat16)
    assert saved_head.dtype == expected_head.dtype
    assert torch.equal(saved_head, expected_head)
    assert source_weights[head_name].dtype == torch.float32

    restored_weights = {
        embedding_name: torch.nn.Parameter(torch.zeros_like(source_weights[embedding_name])),
        head_name: torch.nn.Parameter(torch.zeros_like(source_weights[head_name])),
    }
    loaded_config = SimpleNamespace(**saved_config)
    if nested_text_config:
        loaded_config.text_config = SimpleNamespace(**saved_config["text_config"])
    load_model_hf_weights(
        args,
        [SimpleNamespace(weights=restored_weights)],
        output,
        loaded_config,
        qwen3_5_hf_tensor,
    )
    assert torch.equal(restored_weights[embedding_name], source_weights[embedding_name])
    assert torch.equal(restored_weights[head_name], expected_head.float())
    assert not torch.equal(restored_weights[head_name], restored_weights[embedding_name].float())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
