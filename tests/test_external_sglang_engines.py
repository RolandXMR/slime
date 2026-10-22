import sys
import types
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.backends.sglang_utils.external import (
    ExternalEngineInfo,
    apply_external_engine_info_to_args,
    discover_external_engines,
    start_external_rollout_servers,
    validate_external_fp32_lm_head,
)
from slime.utils.http_utils import get_rollout_num_engines

NUM_GPUS = 0


class _Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_discover_external_engines_reads_server_info(monkeypatch):
    def fake_get(url, timeout):
        assert timeout == 30.0
        assert url == "http://host1:10090/server_info"
        return _Response(
            {
                "tp_size": 4,
                "pp_size": 2,
                "dp_size": 1,
                "ep_size": 4,
                "disaggregation_mode": "null",
            }
        )

    monkeypatch.setattr("slime.backends.sglang_utils.external.requests.get", fake_get)

    infos = discover_external_engines(["host1:10090"])

    assert len(infos) == 1
    info = infos[0]
    assert info.url == "http://host1:10090"
    assert info.host == "host1"
    assert info.port == 10090
    assert info.worker_type == "regular"
    assert info.num_gpus == 8
    assert info.server_info["tp_size"] == 4
    assert info.server_info["pp_size"] == 2
    assert info.server_info["dp_size"] == 1
    assert info.server_info["ep_size"] == 4
    assert info.parallel_config == {"tp_size": 4, "pp_size": 2, "ep_size": 4, "moe_dp_size": 1}


def test_start_external_rollout_servers_exposes_parallel_configs(monkeypatch):
    class FakeActor:
        init = Namespace(remote=lambda **kwargs: kwargs)

    class FakeActorClass:
        def options(self, **kwargs):
            return self

        def remote(self, **kwargs):
            return FakeActor()

    ray = types.ModuleType("ray")
    ray.remote = lambda actor_class: FakeActorClass()
    sglang_engine = types.ModuleType("slime.backends.sglang_utils.sglang_engine")
    sglang_engine.SGLangEngine = object
    ray_utils = types.ModuleType("slime.ray.utils")
    ray_utils.add_default_ray_env_vars = lambda: {}
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.sglang_engine", sglang_engine)
    monkeypatch.setitem(sys.modules, "slime.ray.utils", ray_utils)

    info = ExternalEngineInfo(
        url="http://host1:10090",
        host="host1",
        port=10090,
        worker_type="regular",
        num_gpus=8,
        server_info={"tp_size": 4, "pp_size": 2, "ep_size": 4, "moe_dp_size": 2},
    )
    args = Namespace(rollout_external_engine_infos=[info.to_dict()])

    servers, init_handles = start_external_rollout_servers(args, start_router=lambda *args, **kwargs: ("host1", 30000))

    assert servers["default"].engine_parallel_configs == [{"tp_size": 4, "pp_size": 2, "ep_size": 4, "moe_dp_size": 2}]
    assert len(init_handles) == 1


def test_apply_external_engine_info_handles_pd(monkeypatch):
    payloads = {
        "http://prefill:10090/server_info": {
            "tp_size": 2,
            "pp_size": 1,
            "dp_size": 1,
            "ep_size": 1,
            "disaggregation_mode": "prefill",
            "disaggregation_bootstrap_port": 12090,
        },
        "http://decode:10091/server_info": {
            "tp_size": 4,
            "pp_size": 1,
            "dp_size": 2,
            "ep_size": 2,
            "disaggregation_mode": "decode",
        },
    }

    def fake_get(url, timeout):
        return _Response(payloads[url])

    monkeypatch.setattr("slime.backends.sglang_utils.external.requests.get", fake_get)
    args = Namespace(
        rollout_external=True,
        rollout_external_engine_addrs=["prefill:10090", "decode:10091"],
        rollout_num_gpus=None,
        rollout_num_gpus_per_engine=1,
        sglang_pipeline_parallel_size=1,
        sglang_data_parallel_size=1,
        sglang_expert_parallel_size=1,
        sglang_enable_dp_attention=False,
        router_pd_disaggregation=False,
    )

    apply_external_engine_info_to_args(args)

    assert args.rollout_external is True
    assert args.router_pd_disaggregation is False
    assert args.rollout_num_gpus == 6
    assert args.rollout_num_engines == 2
    assert get_rollout_num_engines(args) == 2
    assert [info["worker_type"] for info in args.rollout_external_engine_infos] == ["prefill", "decode"]
    assert [info["num_gpus"] for info in args.rollout_external_engine_infos] == [2, 4]
    assert [info["server_info"]["dp_size"] for info in args.rollout_external_engine_infos] == [1, 2]
    assert args.rollout_external_engine_infos[0]["disaggregation_bootstrap_port"] == 12090


def test_apply_external_engine_info_preserves_router_pd_flag(monkeypatch):
    def fake_get(url, timeout):
        assert url == "http://regular:10090/server_info"
        return _Response(
            {
                "tp_size": 2,
                "pp_size": 1,
                "disaggregation_mode": "null",
            }
        )

    monkeypatch.setattr("slime.backends.sglang_utils.external.requests.get", fake_get)
    args = Namespace(
        rollout_external=True,
        rollout_external_engine_addrs=["regular:10090"],
        router_pd_disaggregation=True,
    )

    apply_external_engine_info_to_args(args)

    assert args.rollout_external is True
    assert args.router_pd_disaggregation is True
    assert args.rollout_num_gpus == 2
    assert args.rollout_num_engines == 1


def test_apply_external_engine_info_requires_addrs():
    args = Namespace(rollout_external_engine_addrs=None)

    with pytest.raises(ValueError, match="rollout-external-engine-addrs"):
        apply_external_engine_info_to_args(args)


@pytest.fixture
def external_head_config_api(monkeypatch):
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


def test_uses_actual_server_config_and_overrides(external_head_config_api):
    _, calls = external_head_config_api
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
def test_validates_effective_text_head_tying(external_head_config_api, nested, tied):
    api, _ = external_head_config_api
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
def test_unverifiable_config_has_relaunch_guidance(external_head_config_api, info):
    with pytest.raises(ValueError, match="Cannot verify.*Relaunch the external SGLang server"):
        validate_external_fp32_lm_head(info)


def test_missing_tying_field_is_unverifiable(external_head_config_api):
    api, _ = external_head_config_api
    api.config = SimpleNamespace()
    with pytest.raises(ValueError, match="does not report tie_word_embeddings"):
        validate_external_fp32_lm_head({"model_path": "model"})


def test_inaccessible_server_model_has_actionable_error(external_head_config_api):
    api, _ = external_head_config_api

    def fail(*args, **kwargs):
        raise FileNotFoundError("server model is not mounted here")

    api.get_config = fail
    with pytest.raises(ValueError, match="configuration is accessible.*Relaunch"):
        validate_external_fp32_lm_head({"model_path": "/remote/model"})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
