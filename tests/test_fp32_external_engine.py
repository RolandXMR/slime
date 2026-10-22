"""External engines must allocate an independent head before receiving updates."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from slime.backends.sglang_utils.external import validate_external_fp32_lm_head

NUM_GPUS = 0


@pytest.fixture
def config_api(monkeypatch):
    api = ModuleType("sglang.srt.utils.hf_transformers_utils")
    calls = []

    def get_config(model_path, **kwargs):
        calls.append((model_path, kwargs))
        return api.config

    api.config = SimpleNamespace(tie_word_embeddings=False)
    api.get_config = get_config
    api.get_hf_text_config = lambda config: getattr(config, "text_config", config)
    monkeypatch.setitem(sys.modules, api.__name__, api)
    return api, calls


def test_uses_actual_server_config_and_overrides(config_api):
    _, calls = config_api
    info = {
        "model_path": "/actual/server/model",
        "json_model_override_args": '{"tie_word_embeddings":false,"text_config":{"tie_word_embeddings":false}}',
        "trust_remote_code": True,
        "revision": "server-revision",
        "model_config_parser": "hf",
    }
    validate_external_fp32_lm_head(info)
    assert calls == [
        (
            "/actual/server/model",
            {
                "trust_remote_code": True,
                "revision": "server-revision",
                "model_override_args": {
                    "tie_word_embeddings": False,
                    "text_config": {"tie_word_embeddings": False},
                },
                "model_config_parser": "hf",
            },
        )
    ]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_validates_effective_text_head_tying(config_api, nested, tied):
    api, _ = config_api
    text_config = SimpleNamespace(tie_word_embeddings=tied)
    api.config = SimpleNamespace(tie_word_embeddings=not tied, text_config=text_config) if nested else text_config
    if tied:
        with pytest.raises(ValueError, match="external SGLang server with tied embeddings"):
            validate_external_fp32_lm_head({"model_path": "model"})
    else:
        validate_external_fp32_lm_head({"model_path": "model"})


@pytest.mark.parametrize("info", [{}, {"encoder_only": True}])
def test_encoder_workers_need_no_lm_head_config(info):
    validate_external_fp32_lm_head(info, worker_type="encoder" if not info else "regular")


@pytest.mark.parametrize(
    "info",
    [
        {},
        {"model_path": "model", "json_model_override_args": "broken"},
        {"model_path": "model", "json_model_override_args": "[]"},
    ],
)
def test_unverifiable_config_has_relaunch_guidance(config_api, info):
    with pytest.raises(ValueError, match="Cannot verify.*Relaunch the external SGLang server"):
        validate_external_fp32_lm_head(info)


def test_missing_tying_field_is_unverifiable(config_api):
    api, _ = config_api
    api.config = SimpleNamespace()
    with pytest.raises(ValueError, match="does not report tie_word_embeddings"):
        validate_external_fp32_lm_head({"model_path": "model"})


def test_inaccessible_server_model_has_actionable_error(config_api):
    api, _ = config_api

    def fail(*args, **kwargs):
        raise FileNotFoundError("server model is not mounted here")

    api.get_config = fail
    with pytest.raises(ValueError, match="configuration is accessible.*Relaunch"):
        validate_external_fp32_lm_head({"model_path": "/remote/model"})
