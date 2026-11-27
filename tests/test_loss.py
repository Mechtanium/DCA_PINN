"""The loss compares like with like: rates with rates, one weight per residual.

Before 0.2.0 the predicted rate, shape (S, B, 1), was broadcast against six
target columns (q, Di, b, Dinf, n, T), shape (S, B, 6): the rate was fitted
to the mean of a rate and five decline parameters.
"""

import asyncio
import contextlib
import io
import warnings

import pytest
import torch

import workflow as wf

S, B = 8, 3


def tensors(seed=0):
    gen = torch.Generator().manual_seed(seed)
    predicted = torch.rand(S, B, generator=gen)
    observed = torch.rand(S, B, generator=gen)
    weights = torch.softmax(torch.randn(5, generator=gen), dim=0)
    errors = torch.rand(5, generator=gen)
    return predicted, observed, weights, errors


def test_the_data_term_is_the_rate_error():
    predicted, observed, weights, errors = tensors()
    loss = wf.PINNLoss(lambda_data=1.0, lambda_pde=0.0)(predicted, observed, weights, errors)
    torch.testing.assert_close(loss, ((predicted - observed) ** 2).mean())


def test_the_physics_term_is_the_weighted_residual_errors():
    predicted, observed, weights, errors = tensors()
    loss = wf.PINNLoss(lambda_data=0.0, lambda_pde=1.0)(predicted, observed, weights, errors)
    torch.testing.assert_close(loss, (weights * errors).sum())


def test_the_loss_raises_no_broadcast_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wf.PINNLoss()(*tensors())


@pytest.mark.parametrize("case", ["a column of rates against a table", "one weight for five residuals"])
def test_mismatched_shapes_are_refused(case):
    predicted, observed, weights, errors = tensors()
    if case == "a column of rates against a table":
        predicted, observed, match = predicted.unsqueeze(-1), observed.unsqueeze(-1).expand(S, B, 6), "predicted q"
    else:
        weights, match = weights[:1], "model weights"
    with pytest.raises(ValueError, match=match):
        wf.PINNLoss()(predicted, observed, weights, errors)


async def _run(rows, **kwargs):
    async def stream():
        for row in rows:
            yield row

    with contextlib.redirect_stdout(io.StringIO()):
        return [item async for item in wf.train(stream(), **kwargs)]


def test_train_runs_without_warnings():
    _, rows = wf.synthetic_declines("hyperbolic", wells=2, steps=8, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        outputs = asyncio.run(_run(rows, batch_size=2, epochs=2, report_every=1, d_model=16, num_heads=2))
    assert [item[0] for item in outputs] == [1, 2]


def test_train_does_not_read_the_parameter_columns():
    _, rows = wf.synthetic_declines("harmonic", wells=2, steps=8, seed=0)
    blanked = [(t, q, 0.0, 0.0, 0.0, 0.0, 0.0) for t, q, *_ in rows]
    runs = []
    for stream_rows in (rows, blanked):
        torch.manual_seed(0)
        runs.append(asyncio.run(_run(stream_rows, batch_size=2, epochs=3, report_every=1, d_model=16, num_heads=2)))
    assert runs[0] == runs[1]


@pytest.mark.parametrize("kwargs", [{"output_dim": 1}, {"num_pde_constraints": 1}])
def test_train_refuses_shapes_it_cannot_compare(kwargs):
    _, rows = wf.synthetic_declines("exponential", wells=2, steps=4, seed=0)
    with pytest.raises(ValueError, match="output_dim must be 6"):
        asyncio.run(_run(rows, batch_size=2, epochs=1, **kwargs))
