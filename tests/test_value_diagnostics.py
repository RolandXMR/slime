"""Unit tests for ``store_value_diagnostics`` (per-sample critic diagnostics).

CP=1 path only: with CP the function all-gathers first and then runs the same
per-sample math on the full tensors, so the arithmetic pinned here is the
whole contract. Uses the megatron stub from ``_cp_dist_helpers``.
"""

from __future__ import annotations

# Import the helpers BEFORE the slime imports so the megatron stub lands
# in sys.modules first (see test_metric_report.py).
import _cp_dist_helpers  # noqa: F401
import pytest
import torch
from megatron.core import mpu


@pytest.fixture
def cp_disabled(monkeypatch):
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "is_pipeline_last_stage", lambda: True, raising=False)


@pytest.mark.unit
def test_store_value_diagnostics_columns(cp_disabled):
    from slime.backends.megatron_utils.loss import store_value_diagnostics

    rollout_data = {}
    values = [torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([5.0, 6.0])]
    returns = [torch.tensor([1.5, 1.0, 3.5, 3.0]), torch.tensor([5.0, 8.0])]
    # Sample 0 has a masked (tool-observation) token; sample 1's first token is masked.
    loss_masks = [torch.tensor([1.0, 0.0, 1.0, 1.0]), torch.tensor([0.0, 1.0])]
    store_value_diagnostics(rollout_data, [8, 6], [4, 2], values, returns, loss_masks)

    assert rollout_data["value_terminal"] == [4.0, 6.0]
    assert rollout_data["value_terminal_residual"] == [1.0, -2.0]
    # First trainable token, not first response token: sample 1 starts at index 1.
    assert rollout_data["value_first_residual"] == [-0.5, -2.0]
    # Sample 0 masked residuals [-0.5, -0.5, 1.0]: mean 0, |·| mean 2/3, std sqrt(1/2).
    assert rollout_data["value_residual"][0] == pytest.approx(0.0)
    assert rollout_data["value_residual_abs"][0] == pytest.approx(2.0 / 3.0)
    assert rollout_data["value_residual_std"][0] == pytest.approx(0.5**0.5)
    # A single trainable token has no spread.
    assert rollout_data["value_residual_std"][1] == 0.0


@pytest.mark.unit
def test_store_value_diagnostics_all_masked_sample_stays_aligned(cp_disabled):
    from slime.backends.megatron_utils.loss import store_value_diagnostics

    rollout_data = {}
    store_value_diagnostics(
        rollout_data,
        [4, 5],
        [2, 3],
        [torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0, 3.0])],
        [torch.tensor([9.0, 9.0]), torch.tensor([1.0, 2.0, 4.0])],
        [torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0, 1.0])],
    )
    # The fully-masked sample contributes zeros, keeping every column per-sample aligned.
    assert rollout_data["value_terminal"] == [0.0, 3.0]
    assert rollout_data["value_terminal_residual"] == [0.0, -1.0]
    assert all(len(column) == 2 for column in rollout_data.values())


@pytest.mark.unit
def test_custom_advantage_path_stores_value_diagnostics(cp_disabled, monkeypatch):
    from slime.backends.megatron_utils import loss as loss_module

    def custom_advantage(_args, rollout_data):
        rollout_data["advantages"] = [torch.tensor([0.5, -0.5])]
        rollout_data["returns"] = [torch.tensor([1.5, 1.0])]

    monkeypatch.setattr(loss_module, "load_function", lambda _path: custom_advantage)
    rollout_data = {
        "log_probs": [torch.tensor([-0.2, -0.3])],
        "ref_log_probs": None,
        "rewards": [1.0],
        "values": [torch.tensor([1.0, 2.0])],
        "response_lengths": [2],
        "total_lengths": [4],
        "loss_masks": [torch.ones(2)],
    }
    args = type(
        "Args",
        (),
        {
            "use_rollout_logprobs": False,
            "kl_coef": 0.0,
            "kl_loss_type": "k1",
            "custom_advantage_function_path": "custom",
            "use_opd": False,
            "normalize_advantages": False,
        },
    )()

    loss_module.compute_advantages_and_returns(args, rollout_data)

    assert rollout_data["value_terminal"] == [2.0]
    assert rollout_data["value_terminal_residual"] == [1.0]
    assert rollout_data["value_residual"] == pytest.approx([0.25])
