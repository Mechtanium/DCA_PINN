"""The data loss compares the predicted rate with the observed rate.

Before the fix the predicted q, shape (S, B, 1), was broadcast against all
six target columns (q, Di, b, Dinf, n, T), shape (S, B, 6): the rate was
fitted to the mean of a rate and five decline parameters.
"""

import asyncio
import warnings

import pytest
import torch

import workflow as wf

S, B = 8, 3


def tensors(seed=0):
    gen = torch.Generator().manual_seed(seed)
    predicted = torch.randn(S, B, 6, generator=gen)
    observed = torch.randn(S, B, 6, generator=gen)
    weights = torch.rand(S, B, 5, generator=gen)
    residuals = torch.randn(S, B, 5, generator=gen)
    return predicted, observed, weights, residuals


def test_the_data_loss_is_the_rate_error_alone():
    predicted, observed, weights, residuals = tensors()
    loss = wf.PINNLoss(lambda_data=1.0, lambda_pde=0.0)(predicted, observed, weights, residuals)
    expected = ((predicted[..., 0] - observed[..., 0]) ** 2).mean()
    torch.testing.assert_close(loss, expected)


def test_the_parameter_columns_do_not_enter_the_data_loss():
    predicted, observed, weights, residuals = tensors()
    other = observed.clone()
    other[..., 1:] = torch.randn(S, B, 5) * 100
    loss_fn = wf.PINNLoss(lambda_data=1.0, lambda_pde=0.0)
    torch.testing.assert_close(
        loss_fn(predicted, observed, weights, residuals),
        loss_fn(predicted, other, weights, residuals),
    )


def test_the_physics_loss_is_the_weighted_squared_residuals():
    predicted, observed, weights, residuals = tensors()
    loss = wf.PINNLoss(lambda_data=0.0, lambda_pde=1.0)(predicted, observed, weights, residuals)
    torch.testing.assert_close(loss, (weights * residuals**2).sum(dim=-1).mean())


def test_the_loss_raises_no_broadcast_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wf.PINNLoss()(*tensors())


@pytest.mark.parametrize(
    "case",
    ["observed sequence too short", "one weight for five residuals"],
)
def test_mismatched_shapes_are_refused(case):
    predicted, observed, weights, residuals = tensors()
    if case == "observed sequence too short":
        observed = observed[:-1]
        match = "predicted q"
    else:
        weights = weights[..., :1]
        match = "adaptive weights"
    with pytest.raises(ValueError, match=match):
        wf.PINNLoss()(predicted, observed, weights, residuals)


async def _run(rows, **kwargs):
    async def stream():
        for row in rows:
            yield row

    return [item async for item in wf.train(stream(), **kwargs)]


def test_train_compares_like_with_like_without_warnings():
    rows = [(0.1 * i, 1.0 - 0.05 * i, 0.5, 0.5, 1e-3, 0.5, 1.0) for i in range(16)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        outputs = asyncio.run(_run(rows, batch_size=2, epochs=2, report_every=1, d_model=16, num_heads=2))
    assert [epoch for epoch, _ in outputs] == [1, 2]


@pytest.mark.parametrize("kwargs", [{"output_dim": 1}, {"num_pde_constraints": 1}])
def test_train_refuses_shapes_it_cannot_compare(kwargs):
    rows = [(0.1 * i, 1.0, 0.5, 0.5, 1e-3, 0.5, 1.0) for i in range(8)]
    with pytest.raises(ValueError, match="output_dim must be 6"):
        asyncio.run(_run(rows, batch_size=2, epochs=1, **kwargs))
