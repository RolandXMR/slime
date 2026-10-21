import sys
import types
from argparse import Namespace

import pytest
import torch

from slime.utils.ppo_utils import compute_approx_kl

NUM_GPUS = 0


def test_ppo_estimator_does_not_corrupt_logged_kl(monkeypatch):
    import slime.backends.megatron_utils as megatron_utils

    previous_loss = sys.modules.pop("slime.backends.megatron_utils.loss", None)
    previous_cp_utils = sys.modules.pop("slime.backends.megatron_utils.cp_utils", None)
    previous_loss_attr = getattr(megatron_utils, "loss", None)
    previous_cp_utils_attr = getattr(megatron_utils, "cp_utils", None)

    mpu_stub = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        is_pipeline_last_stage=lambda: True,
    )
    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    core_mod.mpu = mpu_stub
    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)

    try:
        from slime.backends.megatron_utils.loss import compute_advantages_and_returns

        log_probs = [torch.tensor([0.5, 0.7, 0.9])]
        ref_log_probs = [torch.tensor([0.4, 0.5, 0.6])]
        expected_kl = compute_approx_kl(log_probs[0], ref_log_probs[0], kl_loss_type="k1")
        rollout_data = {
            "log_probs": log_probs,
            "ref_log_probs": ref_log_probs,
            "rewards": [1.0],
            "values": [torch.zeros(3)],
            "response_lengths": [3],
            "total_lengths": [5],
            "loss_masks": [torch.ones(3)],
        }
        args = Namespace(
            advantage_estimator="ppo",
            kl_coef=0.05,
            kl_loss_type="k1",
            use_rollout_logprobs=False,
            custom_advantage_function_path=None,
            normalize_advantages=False,
            use_opd=False,
            gamma=1.0,
            lambd=1.0,
        )
        compute_advantages_and_returns(args, rollout_data)
        torch.testing.assert_close(rollout_data["kl"][0], expected_kl)
    finally:
        if previous_loss is None:
            sys.modules.pop("slime.backends.megatron_utils.loss", None)
        else:
            sys.modules["slime.backends.megatron_utils.loss"] = previous_loss
        if previous_cp_utils is None:
            sys.modules.pop("slime.backends.megatron_utils.cp_utils", None)
        else:
            sys.modules["slime.backends.megatron_utils.cp_utils"] = previous_cp_utils
        if previous_loss_attr is None:
            megatron_utils.__dict__.pop("loss", None)
        else:
            megatron_utils.loss = previous_loss_attr
        if previous_cp_utils_attr is None:
            megatron_utils.__dict__.pop("cp_utils", None)
        else:
            megatron_utils.cp_utils = previous_cp_utils_attr


def test_policy_loss_reports_train_rollout_kl(monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    current = [torch.tensor([-0.2, -0.4], requires_grad=True)]
    old = [torch.tensor([-0.3, -0.5])]
    rollout = [torch.tensor([-0.35, -0.45])]
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *_args, **_kwargs: (None, {"log_probs": current, "entropy": [torch.zeros(2)]}),
    )
    args = Namespace(
        use_rollout_logprobs=False,
        rollout_top_p=1.0,
        advantage_estimator="ppo",
        use_opsm=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=None,
        get_mismatch_metrics=False,
        use_tis=False,
        custom_pg_loss_reducer_function_path=None,
        entropy_coef=0.0,
        use_kl_loss=False,
        use_opd=False,
    )
    batch = {
        "advantages": [torch.ones(2)],
        "log_probs": old,
        "rollout_log_probs": rollout,
        "unconcat_tokens": [torch.tensor([1, 2, 3, 4])],
        "response_lengths": [2],
        "total_lengths": [4],
        "loss_masks": [torch.ones(2)],
        "rollout_mask_sums": torch.tensor([2.0]),
    }

    def reducer(value):
        return value.mean()

    _, metrics = loss_module.policy_loss_function(args, batch, torch.zeros(1, 1, 1), reducer)

    expected = compute_approx_kl(rollout[0], old[0], kl_loss_type="low_var_kl").mean()
    torch.testing.assert_close(metrics["train_rollout_kl"], expected)
    assert {"_policy_stat_count", "_policy_stat_is_sum", "_policy_stat_is_sq_sum"} <= metrics.keys()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
