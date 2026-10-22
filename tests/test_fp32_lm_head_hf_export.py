"""Saved heads survive a real safetensors round trip independently of embeddings."""

import importlib.util
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from slime.backends.megatron_utils.hf_checkpoint_saver import save_hf_model_to_path
from slime.backends.megatron_utils.hf_to_megatron.common import load_model_hf_weights
from slime.backends.megatron_utils.hf_to_megatron.qwen3_5 import qwen3_5_hf_tensor
from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf

NUM_GPUS = 0


def _args(source):
    return SimpleNamespace(
        hf_checkpoint=str(source),
        fp32_lm_head=True,
        vocab_size=4,
        padded_vocab_size=4,
        q_lora_rank=None,
    )


def test_default_rollout_conversion_keeps_actor_head_fp32(tmp_path):
    head = torch.full((4, 3), 1.001, dtype=torch.float32)
    converted = convert_to_hf(_args(tmp_path), "qwen3_5", "module.module.output_layer.weight", head)

    assert converted[0][0] == "lm_head.weight"
    assert converted[0][1].dtype == torch.bfloat16
    assert torch.equal(converted[0][1], head.to(torch.bfloat16))
    assert head.dtype == torch.float32
    assert not torch.equal(converted[0][1].float(), head)


@pytest.mark.parametrize("script_name", ["convert_torch_dist_to_hf.py", "convert_torch_dist_to_hf_parallel.py"])
def test_checkpoint_conversion_tools_preserve_fp32_head_and_untied_config(tmp_path, script_name):
    path = Path(__file__).resolve().parents[1] / "tools" / script_name
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
    state = {
        "embedding.word_embeddings.weight": torch.full((4, 3), 2.0, dtype=torch.bfloat16),
        "output_layer.weight": torch.full((4, 3), 1.001, dtype=torch.float32),
    }
    converter.save_tensors(_args(source), "qwen3_5", state, str(output), chunk_size=1024)
    converter.copy_assets(str(source), str(output), untie_embeddings=True)

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
    args = _args(source)

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
