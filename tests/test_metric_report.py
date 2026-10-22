"""Single-process metric-report invariance tests.

Pins train-side / rollout-side report formulas implemented in
``slime.observability.train_metric_utils``: the reported number for a given set
of samples must be the same regardless of

  - how samples are distributed across micro-batches / DP ranks
  - whether context parallelism is on or off
  - whether the path is per-rollout-mean or per-token-loss

Single-process variants use a mock dp-with-cp group + a no-op
``dist.all_reduce`` to keep things lightweight; the multi-process
end-to-end variants (real torch.distributed) live in
``test_metric_report_dist.py``.
"""

from __future__ import annotations

from argparse import Namespace

# Import the helpers BEFORE the slime imports so the megatron stub lands
# in sys.modules first. pytest's prepend importmode puts this file's
# directory (``tests/``) on sys.path, which is what makes the bare-name
# import work without an ``__init__.py``.
import _cp_dist_helpers  # noqa: F401
import pytest
import torch

from slime.backends.megatron_utils.cp_utils import (  # noqa: E402
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
)
from slime.observability.train_metric_utils import (  # noqa: E402
    reduce_train_step_metrics,
    rollout_log_metric_contribution,
)
from slime.utils.ppo_utils import compute_approx_kl

NUM_GPUS = 0


def test_truncation_fraction_is_reported_once_for_all_rollout_sources(monkeypatch):
    from types import SimpleNamespace

    from megatron.core import mpu

    from slime.observability import train_metric_utils
    from slime.observability.rollout_metrics import compute_metrics_from_samples
    from slime.utils.types import Sample

    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(mpu, "is_pipeline_last_stage", lambda: True, raising=False)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mpu, "get_data_parallel_world_size", lambda **kwargs: 1, raising=False)
    captured = {}
    monkeypatch.setattr(
        train_metric_utils, "gather_log_data", lambda name, args, step, metrics: captured.update(metrics)
    )
    args = SimpleNamespace(
        ci_test=False,
        log_multi_turn=False,
        log_passrate=False,
        log_correct_samples=False,
        advantage_estimator="ppo",
        log_reward_category=None,
    )
    samples = [
        Sample(index=0, response_length=1, response="ok", status=Sample.Status.COMPLETED),
        Sample(index=1, response_length=1, response="partial", status=Sample.Status.TRUNCATED),
    ]
    rollout_metrics = compute_metrics_from_samples(args, samples)
    assert "truncated_ratio" not in rollout_metrics
    assert "truncated" not in rollout_metrics
    data = {
        "response_lengths": [1, 1],
        "total_lengths": [2, 2],
        "loss_masks": [[1], [1]],
        "global_batch_sizes": [2],
        "truncated": [0, 1],
    }
    train_metric_utils.log_rollout_data(0, args, data)
    assert captured["truncated"] == (1, 2)
    assert "truncated_ratio" not in captured
    assert captured["response_lengths"] == (2, 2)
    assert data["truncated"] == [0, 1]  # Training still receives the internal flags.


@pytest.fixture
def mock_dp_with_cp_group(monkeypatch):
    """A sentinel "process group" object plus a no-op ``dist.all_reduce``.

    Lets the train-step report tests exercise the production call shape
    (``dist.all_reduce(values, group=dp_with_cp_group)``) without standing
    up a real torch.distributed runtime. The test itself simulates cross-
    rank summation in pure Python and feeds the already-summed tensor in;
    the no-op all_reduce leaves it untouched.
    """
    import torch.distributed as dist

    monkeypatch.setattr(dist, "all_reduce", lambda tensor, group=None, op=None: None)
    return object()  # opaque sentinel — only used as the ``group`` argument


# ---------------------------------------------------------------------------
# Mirrors the actual train_one_step reporting math:
#
#   per-rollout-mean path:
#       reported = sum_of_reducer_per_mb / step_global_batch_size
#   per-token-loss path:
#       reported = sum_of_reducer_per_mb / sum_of_per_mb_num_tokens
#
# The reducer is the same callable used at train time (and inside
# log_rollout_data on the rollout side).
# ---------------------------------------------------------------------------


# 4 samples: rollout R0 owns indices 0,1,2 (mask sums 3+3+3=9); rollout R1
# owns index 3 (mask sum 3). Pre-computed per-sample denom = group sum.
# Per-rollout-mean: R0 = 5, R1 = 11, sum = 16, divided by 2 rollouts → 8.
# Per-token-loss:   sum of all x = 78, total clamped mask = 12, → 6.5.
_FIXED_RESPONSE_LENGTHS = [3, 3, 3, 3]
_FIXED_TOTAL_LENGTHS = [r + 4 for r in _FIXED_RESPONSE_LENGTHS]
_FIXED_LOSS_MASKS = [torch.ones(r, dtype=torch.float32) for r in _FIXED_RESPONSE_LENGTHS]
_FIXED_ROLLOUT_DENOMS = [9.0, 9.0, 9.0, 3.0]
_FIXED_X_PER_SAMPLE = [
    torch.tensor([1.0, 2.0, 3.0]),
    torch.tensor([4.0, 5.0, 6.0]),
    torch.tensor([7.0, 8.0, 9.0]),
    torch.tensor([10.0, 11.0, 12.0]),
]
_FIXED_STEP_GBS = 2  # 2 distinct rollouts in the step
_EXPECTED_PER_ROLLOUT_MEAN_REPORT = 8.0
_EXPECTED_PER_TOKEN_LOSS_REPORT = 78.0 / 12.0


# Each entry: list of "rank"s, each rank is a list of mbs, each mb is the
# sample-index list packed into that mb. Covers: single mb, evenly split by
# rollout, split inside a rollout (R0 across mbs), uneven distribution, and
# fully singleton mbs per rank.
_PARTITION_CONFIGS = [
    [[[0, 1, 2, 3]]],  # 1 rank, 1 mb
    [[[0, 1, 2], [3]]],  # 1 rank, 2 mbs split at rollout boundary
    [[[0, 1], [2, 3]]],  # 1 rank, 2 mbs splitting R0 across them — the tricky case
    [[[0, 1]], [[2, 3]]],  # 2 ranks, 1 mb each
    [[[0, 1, 3]], [[2]]],  # 2 ranks, R0 split across BOTH ranks (worst case for split-across-mb bug)
    [[[0]], [[1]], [[2]], [[3]]],  # 4 ranks, 1 sample per rank
]


def _simulate_report(partition, *, per_token_loss: bool) -> float:
    """Reproduce train_one_step's reporting math for one partition config."""
    metric_sum = 0.0
    num_tokens_sum = 0
    for rank_mbs in partition:
        for mb_indices in rank_mbs:
            mb_total = [_FIXED_TOTAL_LENGTHS[i] for i in mb_indices]
            mb_resp = [_FIXED_RESPONSE_LENGTHS[i] for i in mb_indices]
            mb_masks = [_FIXED_LOSS_MASKS[i] for i in mb_indices]
            mb_x = torch.cat([_FIXED_X_PER_SAMPLE[i] for i in mb_indices])
            if per_token_loss:
                # Per-token-loss: caller uses ``calculate_per_token_loss=True``
                # to get ``sum_of_token`` (no per-sample denom).
                reducer = get_sum_of_sample_mean(mb_total, mb_resp, mb_masks, calculate_per_token_loss=True)
                num_tokens_sum += sum(max(int(m.sum().item()), 1) for m in mb_masks)
            else:
                mb_denoms = torch.tensor([_FIXED_ROLLOUT_DENOMS[i] for i in mb_indices], dtype=torch.float32)
                reducer = get_sum_of_sample_mean(mb_total, mb_resp, mb_masks, mb_denoms)
            metric_sum += reducer(mb_x).item()
    if per_token_loss:
        return metric_sum / num_tokens_sum
    return metric_sum / _FIXED_STEP_GBS


@pytest.mark.unit
@pytest.mark.parametrize("partition", _PARTITION_CONFIGS)
def test_per_rollout_mean_report_invariant_to_mb_distribution(partition):
    """Same samples should yield the same per-rollout-mean report regardless of
    how they're spread across DP ranks / micro-batches — this is what lets us
    change parallelism without changing wandb numbers."""
    assert _simulate_report(partition, per_token_loss=False) == pytest.approx(_EXPECTED_PER_ROLLOUT_MEAN_REPORT)


@pytest.mark.unit
@pytest.mark.parametrize("partition", _PARTITION_CONFIGS)
def test_per_token_loss_report_invariant_to_mb_distribution(partition):
    """Same invariant for the per-token-loss reporting path."""
    assert _simulate_report(partition, per_token_loss=True) == pytest.approx(_EXPECTED_PER_TOKEN_LOSS_REPORT)


def _simulate_rollout_report(samples_per_rank):
    """Reproduce log_rollout_data + gather_log_data's averaging math for the
    per-token metric branch.

    Each "rank" applies the reducer once over its full sample subset, then
    ``rollout_log_metric_contribution`` (the same helper the reporter uses) emits
    the ``(per_rank_sum, count)`` tuple. We aggregate via
    ``Σsum / Σcount`` — the same shape ``gather_log_data`` uses.
    """
    dp_size = len(samples_per_rank)
    pairs: list[tuple[float, float]] = []
    for indices in samples_per_rank:
        if not indices:
            pairs.append(
                rollout_log_metric_contribution(
                    0.0, cp_size=1, num_rollouts_in_rollout=_FIXED_STEP_GBS, dp_size=dp_size
                )
            )
            continue
        tl = [_FIXED_TOTAL_LENGTHS[i] for i in indices]
        rl = [_FIXED_RESPONSE_LENGTHS[i] for i in indices]
        masks = [_FIXED_LOSS_MASKS[i] for i in indices]
        denoms = torch.tensor([_FIXED_ROLLOUT_DENOMS[i] for i in indices], dtype=torch.float32)
        x = torch.cat([_FIXED_X_PER_SAMPLE[i] for i in indices])
        reducer = get_sum_of_sample_mean(tl, rl, masks, denoms)
        pairs.append(
            rollout_log_metric_contribution(
                reducer(x).item(),
                cp_size=1,
                num_rollouts_in_rollout=_FIXED_STEP_GBS,
                dp_size=dp_size,
            )
        )
    total_sum = sum(p[0] for p in pairs)
    total_count = sum(p[1] for p in pairs)
    return total_sum / total_count


_DP_PARTITIONS = [
    [[0, 1, 2, 3]],  # 1 rank holds everything
    [[0, 1, 2], [3]],  # 2 ranks, balanced by rollout
    [[0, 1], [2, 3]],  # 2 ranks splitting R0 across mb-and-rank
    [[0, 1, 3], [2]],  # 2 ranks with R0 spread across BOTH (one of R0's samples is on rank 1)
    [[0], [1], [2], [3]],  # 4 ranks, one sample each (R0's samples spread across 3 ranks)
]


@pytest.mark.unit
@pytest.mark.parametrize("dp_partition", _DP_PARTITIONS)
def test_rollout_report_matches_train_report_in_single_step(dp_partition):
    """In a 1-step rollout, the rollout-side report (log_rollout_data → gather)
    must equal the train-side report (train_one_step ``value / step_global_batch_size``)
    for the same samples — otherwise wandb numbers between phases drift.

    Both go through the same reducer with the same precomputed denominators;
    the contract this test pins is that the gather count plumbing on the
    rollout side sums to the same denominator the train side uses
    (``step_global_batch_size``), independent of how the rollout's samples
    are spread across DP ranks.
    """
    rollout_report = _simulate_rollout_report(dp_partition)
    assert rollout_report == pytest.approx(_EXPECTED_PER_ROLLOUT_MEAN_REPORT)


@pytest.mark.unit
def test_train_one_step_per_rollout_mean_report_invariant_to_cp(monkeypatch, mock_dp_with_cp_group):
    """End-to-end check of train_one_step's report formula across CP sizes.

    Mirrors the actual reduction order:
      1. Each (DP, CP) rank computes per-mb reducer output.
      2. Per-rank values are summed across mbs locally.
      3. All-reduce sums across DP*CP ranks.
      4. ``reduce_train_step_metrics`` applied (the same helper
         ``train_one_step`` calls, so this test stays honest if the
         implementation changes).

    cp_size = 1 vs cp_size = 2 must give the same reported number —
    otherwise wandb metrics would drift the moment a user enables CP.
    """
    from megatron.core import mpu as _mpu

    total_lengths = [12, 12]
    response_lengths = [8, 8]
    loss_masks = [torch.ones(r, dtype=torch.float32) for r in response_lengths]
    sample_denoms = torch.tensor([16.0, 16.0], dtype=torch.float32)
    x_full = [
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
    ]
    step_global_batch_size = 1  # one rollout in the step

    def simulate(cp_size: int) -> float:
        monkeypatch.setattr(_mpu, "get_context_parallel_world_size", lambda: cp_size)
        # values[0] is the per-token-loss path's num_tokens slot; for
        # per-rollout-mean it's a zero placeholder (loss_function sets 0).
        value_after_allreduce = 0.0
        for cp_rank in range(cp_size):
            monkeypatch.setattr(_mpu, "get_context_parallel_rank", lambda r=cp_rank: r)
            if cp_size == 1:
                x_for_rank = torch.cat(x_full)
            else:
                x_chunks_per_sample = []
                for tl, rl, x in zip(total_lengths, response_lengths, x_full, strict=True):
                    prompt_length = tl - rl
                    _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(tl, rl)
                    c0 = x[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
                    c1 = x[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
                    x_chunks_per_sample.append(torch.cat([c0, c1]))
                x_for_rank = torch.cat(x_chunks_per_sample)
            reducer = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, sample_denoms)
            value_after_allreduce += reducer(x_for_rank).item()
        reduced = reduce_train_step_metrics(
            [{"keys": ["metric"], "values": torch.tensor([0.0, value_after_allreduce])}],
            calculate_per_token_loss=False,
            step_global_batch_size=step_global_batch_size,
            cp_size=cp_size,
            dp_with_cp_group=mock_dp_with_cp_group,
        )
        return reduced["metric"]

    assert simulate(1) == pytest.approx(simulate(2))


@pytest.mark.unit
def test_train_one_step_per_token_loss_report_invariant_to_cp(monkeypatch, mock_dp_with_cp_group):
    """Same end-to-end check for the per-token-loss path: divisor is
    ``values[0] = num_tokens`` (computed in loss.py from FULL loss masks),
    which each CP rank duplicates and all-reduce sums by ``cp_size``. The
    ``cp_factor = cp_size`` multiplier inside ``reduce_train_step_metrics``
    cancels that inflation, so the report stays CP-invariant.
    """
    from megatron.core import mpu as _mpu

    total_lengths = [12, 12]
    response_lengths = [8, 8]
    loss_masks = [torch.ones(r, dtype=torch.float32) for r in response_lengths]
    num_tokens_per_mb = sum(int(m.sum().item()) for m in loss_masks)  # = 16
    x_full = [
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
    ]

    def simulate(cp_size: int) -> float:
        monkeypatch.setattr(_mpu, "get_context_parallel_world_size", lambda: cp_size)
        value_after_allreduce = 0.0
        num_tokens_after_allreduce = 0  # each CP rank reports the same num_tokens
        for cp_rank in range(cp_size):
            monkeypatch.setattr(_mpu, "get_context_parallel_rank", lambda r=cp_rank: r)
            if cp_size == 1:
                x_for_rank = torch.cat(x_full)
            else:
                x_chunks_per_sample = []
                for tl, rl, x in zip(total_lengths, response_lengths, x_full, strict=True):
                    prompt_length = tl - rl
                    _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(tl, rl)
                    c0 = x[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
                    c1 = x[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
                    x_chunks_per_sample.append(torch.cat([c0, c1]))
                x_for_rank = torch.cat(x_chunks_per_sample)
            reducer = get_sum_of_sample_mean(
                total_lengths, response_lengths, loss_masks, calculate_per_token_loss=True
            )
            value_after_allreduce += reducer(x_for_rank).item()
            num_tokens_after_allreduce += num_tokens_per_mb
        reduced = reduce_train_step_metrics(
            [
                {
                    "keys": ["metric"],
                    "values": torch.tensor([num_tokens_after_allreduce, value_after_allreduce], dtype=torch.float32),
                }
            ],
            calculate_per_token_loss=True,
            step_global_batch_size=999,  # unused in per-token-loss path
            cp_size=cp_size,
            dp_with_cp_group=mock_dp_with_cp_group,
        )
        return reduced["metric"]

    assert simulate(1) == pytest.approx(simulate(2))


# ---------------------------------------------------------------------------
# Value-head sufficient statistics: value_loss_function reports raw masked
# sums (count / Σx / Σx²) under ``_value_stat_*`` keys, and
# ``reduce_train_step_metrics`` → ``finalize_value_head_stats`` reconstructs
# explained variance / bias / stds from them after the all-reduce. The tests
# pin that the reconstruction equals a direct full-batch computation, in both
# reporting modes and independent of mb splits.
# ---------------------------------------------------------------------------

_VS_RETURNS = [
    torch.tensor([1.0, 2.0, 3.0]),
    torch.tensor([4.0, 5.0, 6.0]),
    torch.tensor([7.0, 8.0, 9.0]),
    torch.tensor([10.0, 11.0, 12.0]),
]
_VS_VALUES = [
    torch.tensor([1.5, 1.5, 3.5]),
    torch.tensor([3.0, 5.5, 5.0]),
    torch.tensor([8.0, 7.0, 10.0]),
    torch.tensor([9.0, 12.0, 11.5]),
]
# One masked token exercises that the stats honor the loss mask.
_VS_LOSS_MASKS = [
    torch.tensor([1.0, 1.0, 1.0]),
    torch.tensor([1.0, 0.0, 1.0]),
    torch.tensor([1.0, 1.0, 1.0]),
    torch.tensor([1.0, 1.0, 1.0]),
]


def _vs_reference() -> dict[str, float]:
    """Direct full-batch computation of the reconstructed metrics."""
    active = torch.cat(_VS_LOSS_MASKS).bool()
    returns = torch.cat(_VS_RETURNS)[active]
    values = torch.cat(_VS_VALUES)[active]
    residual = values - returns
    return {
        "value_explained_variance": (1.0 - residual.var(correction=0) / returns.var(correction=0)).item(),
        "value_bias": residual.mean().item(),
        "value_std": values.std(correction=0).item(),
        "return_std": returns.std(correction=0).item(),
    }


def _vs_stat_values(mb_indices) -> dict[str, float]:
    """The six sufficient statistics for one micro-batch, via the same reducer
    value_loss_function uses (``calculate_per_token_loss=True`` → raw masked sums)."""
    tl = [_FIXED_TOTAL_LENGTHS[i] for i in mb_indices]
    rl = [3 for _ in mb_indices]
    masks = [_VS_LOSS_MASKS[i] for i in mb_indices]
    returns = torch.cat([_VS_RETURNS[i] for i in mb_indices])
    values = torch.cat([_VS_VALUES[i] for i in mb_indices])
    residual = values - returns
    sum_of_token = get_sum_of_sample_mean(tl, rl, masks, calculate_per_token_loss=True)
    return {
        "_value_stat_count": sum_of_token(torch.ones_like(returns)).item(),
        "_value_stat_return_sum": sum_of_token(returns).item(),
        "_value_stat_return_sq_sum": sum_of_token(returns.square()).item(),
        "_value_stat_residual_sum": sum_of_token(residual).item(),
        "_value_stat_residual_sq_sum": sum_of_token(residual.square()).item(),
        "_value_stat_value_sq_sum": sum_of_token(values.square()).item(),
    }


@pytest.mark.unit
@pytest.mark.parametrize("partition", _PARTITION_CONFIGS)
@pytest.mark.parametrize("per_token_loss", [False, True])
def test_value_stats_reconstruction_matches_direct(partition, per_token_loss, mock_dp_with_cp_group):
    """EV / bias / stds reconstructed from mb-summed sufficient statistics must
    equal the direct full-batch numbers, regardless of the mb/DP split and of
    which reporting mode scaled the sums on their way through."""
    losses_reduced = []
    for rank_mbs in partition:
        for mb_indices in rank_mbs:
            stats = _vs_stat_values(mb_indices)
            # value_loss / value_clipfrac ride along like production does.
            log = {"value_loss": 0.0, **stats}
            num_tokens = sum(max(int(_VS_LOSS_MASKS[i].sum().item()), 1) for i in mb_indices)
            losses_reduced.append(
                {
                    "keys": list(log.keys()),
                    "values": torch.tensor(
                        [num_tokens if per_token_loss else 0.0] + list(log.values()), dtype=torch.float64
                    ),
                }
            )
    reduced = reduce_train_step_metrics(
        losses_reduced,
        calculate_per_token_loss=per_token_loss,
        step_global_batch_size=_FIXED_STEP_GBS,
        cp_size=1,
        dp_with_cp_group=mock_dp_with_cp_group,
    )

    for key, expected in _vs_reference().items():
        assert reduced[key] == pytest.approx(expected, abs=1e-6), key
    assert not any(key.startswith("_value_stat_") for key in reduced)


@pytest.mark.unit
def test_value_stats_reconstruction_invariant_to_cp(monkeypatch, mock_dp_with_cp_group):
    """With CP, each rank contributes the sums of its zigzag chunks; the
    all-reduced totals have the same ratios, even after the common reporting
    normalizer is applied."""
    from megatron.core import mpu as _mpu

    total_lengths = [12, 12]
    response_lengths = [8, 8]
    loss_masks = [
        torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
    ]
    returns = [torch.arange(1.0, 9.0), torch.arange(10.0, 80.0 + 1, 10.0)]
    values = [r + torch.linspace(-1.0, 1.0, 8) for r in returns]

    active = torch.cat(loss_masks).bool()
    flat_returns = torch.cat(returns)[active]
    flat_residual = (torch.cat(values) - torch.cat(returns))[active]
    expected_ev = (1.0 - flat_residual.var(correction=0) / flat_returns.var(correction=0)).item()

    def simulate(cp_size: int) -> float:
        monkeypatch.setattr(_mpu, "get_context_parallel_world_size", lambda: cp_size)
        stat_keys = (
            "_value_stat_count",
            "_value_stat_return_sum",
            "_value_stat_return_sq_sum",
            "_value_stat_residual_sum",
            "_value_stat_residual_sq_sum",
            "_value_stat_value_sq_sum",
        )
        sums = dict.fromkeys(stat_keys, 0.0)
        num_tokens_per_rank = sum(max(int(m.sum().item()), 1) for m in loss_masks)
        num_tokens_after_allreduce = 0
        for cp_rank in range(cp_size):
            monkeypatch.setattr(_mpu, "get_context_parallel_rank", lambda r=cp_rank: r)

            def chunk(x_full):
                if cp_size == 1:
                    return torch.cat(x_full)
                parts = []
                for tl, rl, x in zip(total_lengths, response_lengths, x_full, strict=True):
                    prompt_length = tl - rl
                    _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(tl, rl)
                    c0 = x[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
                    c1 = x[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
                    parts.append(torch.cat([c0, c1]))
                return torch.cat(parts)

            sum_of_token = get_sum_of_sample_mean(
                total_lengths, response_lengths, loss_masks, calculate_per_token_loss=True
            )
            rank_returns = chunk(returns)
            rank_values = chunk(values)
            rank_residual = rank_values - rank_returns
            sums["_value_stat_count"] += sum_of_token(torch.ones_like(rank_returns)).item()
            sums["_value_stat_return_sum"] += sum_of_token(rank_returns).item()
            sums["_value_stat_return_sq_sum"] += sum_of_token(rank_returns.square()).item()
            sums["_value_stat_residual_sum"] += sum_of_token(rank_residual).item()
            sums["_value_stat_residual_sq_sum"] += sum_of_token(rank_residual.square()).item()
            sums["_value_stat_value_sq_sum"] += sum_of_token(rank_values.square()).item()
            num_tokens_after_allreduce += num_tokens_per_rank

        reduced = reduce_train_step_metrics(
            [
                {
                    "keys": list(sums.keys()),
                    "values": torch.tensor([num_tokens_after_allreduce] + list(sums.values()), dtype=torch.float64),
                }
            ],
            calculate_per_token_loss=True,
            step_global_batch_size=999,
            cp_size=cp_size,
            dp_with_cp_group=mock_dp_with_cp_group,
        )
        return reduced["value_explained_variance"]

    assert simulate(1) == pytest.approx(expected_ev, abs=1e-6)
    assert simulate(2) == pytest.approx(expected_ev, abs=1e-6)


@pytest.mark.unit
def test_finalize_value_head_stats_passthrough_without_stats():
    """Policy-loss metrics (no ``_value_stat_*`` keys) must pass through untouched."""
    from slime.observability.train_metric_utils import finalize_value_head_stats

    metrics = {"pg_loss": 0.5, "ppo_kl": 0.01}
    untouched = dict(metrics)
    finalize_value_head_stats(untouched)
    assert untouched == metrics


@pytest.mark.unit
@pytest.mark.parametrize("per_token_loss", [False, True])
def test_ess_ratio_reconstruction_matches_direct(per_token_loss, mock_dp_with_cp_group):
    weights = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
    stats = {
        "_policy_stat_count": float(weights.numel()),
        "_policy_stat_is_sum": weights.sum().item(),
        "_policy_stat_is_sq_sum": weights.square().sum().item(),
    }
    normalizer = weights.numel() if per_token_loss else 2
    reduced = reduce_train_step_metrics(
        [
            {
                "keys": list(stats),
                "values": torch.tensor([normalizer if per_token_loss else 0.0, *stats.values()], dtype=torch.float64),
            }
        ],
        calculate_per_token_loss=per_token_loss,
        step_global_batch_size=normalizer,
        cp_size=1,
        dp_with_cp_group=mock_dp_with_cp_group,
    )

    expected = weights.sum().square() / (weights.numel() * weights.square().sum())
    assert reduced["ess_ratio"] == pytest.approx(expected.item())
    assert not any(key.startswith("_policy_stat_") for key in reduced)


@pytest.mark.unit
def test_ess_ratio_rejects_non_finite_stats():
    from slime.observability.train_metric_utils import finalize_policy_stats

    metrics = {
        "_policy_stat_count": 2.0,
        "_policy_stat_is_sum": float("inf"),
        "_policy_stat_is_sq_sum": float("inf"),
    }
    finalize_policy_stats(metrics)

    assert metrics == {"ess_ratio": 0.0}


@pytest.fixture
def cp_disabled(monkeypatch):
    from megatron.core import mpu

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
@pytest.mark.parametrize("store_diagnostics", [False, True])
def test_custom_advantage_path_stores_value_diagnostics(cp_disabled, monkeypatch, store_diagnostics):
    from slime.backends.megatron_utils import loss as loss_module

    def custom_advantage(_args, rollout_data):
        rollout_data["advantages"] = [torch.tensor([0.5, -0.5])]
        rollout_data["returns"] = [torch.tensor([1.5, 1.0])]

    monkeypatch.setattr(loss_module, "load_function", lambda _path: custom_advantage)
    if not store_diagnostics:
        monkeypatch.setattr(
            loss_module, "store_value_diagnostics", lambda *_: pytest.fail("unused diagnostics must be skipped")
        )
    rollout_data = {
        "log_probs": [torch.tensor([-0.2, -0.3])],
        "ref_log_probs": None,
        "rewards": [1.0],
        "values": [torch.tensor([1.0, 2.0])],
        "response_lengths": [2],
        "total_lengths": [4],
        "loss_masks": [torch.ones(2)],
    }
    args = Namespace(
        use_rollout_logprobs=False,
        kl_coef=0.0,
        kl_loss_type="k1",
        custom_advantage_function_path="custom",
        use_opd=False,
        normalize_advantages=False,
    )

    loss_module.compute_advantages_and_returns(args, rollout_data, store_diagnostics=store_diagnostics)

    torch.testing.assert_close(rollout_data["advantages"][0], torch.tensor([0.5, -0.5]))
    torch.testing.assert_close(rollout_data["returns"][0], torch.tensor([1.5, 1.0]))
    if store_diagnostics:
        assert rollout_data["value_terminal"] == [2.0]
        assert rollout_data["value_terminal_residual"] == [1.0]
        assert rollout_data["value_residual"] == pytest.approx([0.25])
    else:
        assert "value_terminal" not in rollout_data


@pytest.mark.unit
@pytest.mark.parametrize("count", [0.0, 1.0])
def test_value_stats_empty_and_single_active_token(count):
    from slime.observability.train_metric_utils import finalize_value_head_stats

    metrics = {
        "_value_stat_count": count,
        "_value_stat_return_sum": count * 2.0,
        "_value_stat_return_sq_sum": count * 4.0,
        "_value_stat_residual_sum": count,
        "_value_stat_residual_sq_sum": count,
        "_value_stat_value_sq_sum": count * 9.0,
    }
    finalize_value_head_stats(metrics)

    assert metrics == (
        {"value_explained_variance": 0.0, "value_bias": 1.0, "value_std": 0.0, "return_std": 0.0} if count else {}
    )


def test_ppo_estimator_does_not_corrupt_logged_kl(cp_disabled):
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


@pytest.mark.parametrize("log_ratios", [[0.1, 0.1], [25.0, 30.0]])
def test_policy_loss_reports_train_rollout_kl_and_ess(cp_disabled, monkeypatch, log_ratios):
    from slime.backends.megatron_utils import loss as loss_module
    from slime.observability.train_metric_utils import finalize_policy_stats

    current = [torch.tensor([-0.2, -0.4], requires_grad=True)]
    old = [current[0].detach() - torch.tensor(log_ratios)]
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
    stats = {key: value.item() for key, value in metrics.items() if key.startswith("_policy_stat_")}
    finalize_policy_stats(stats)
    normalized_weights = torch.tensor(log_ratios, dtype=torch.float64).softmax(0)
    expected_ess = 1.0 / (len(log_ratios) * normalized_weights.square().sum())
    assert stats["ess_ratio"] == pytest.approx(expected_ess.item())



if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
