"""Training on one model's declines drives that model's weight towards 1.

synthetic_declines draws four wells from one decline model (seeded); train
runs with its defaults (100 epochs) on CPU, about a second per model, and
must end with the generating model's weight above 0.9, the model it names
being that one.
"""

import asyncio
import contextlib
import io
import math
import statistics

import pytest
import torch

import workflow as wf


async def train_outputs(rows, **kwargs):
    async def stream():
        for row in rows:
            yield row

    with contextlib.redirect_stdout(io.StringIO()):
        return [item async for item in wf.train(stream(), **kwargs)]


@pytest.mark.parametrize("model", wf.DECLINE_MODELS)
def test_training_finds_the_generating_model(model):
    torch.manual_seed(0)
    _, rows = wf.synthetic_declines(model, wells=4, steps=32, seed=0)
    outputs = asyncio.run(train_outputs(rows, batch_size=4, report_every=1))
    assert len(outputs) == 100
    epoch, loss, chosen, *weights = outputs[-1]
    assert chosen == model
    assert weights[wf.DECLINE_MODELS.index(model)] > 0.9, dict(zip(wf.DECLINE_MODELS, weights))
    assert sum(weights) == pytest.approx(1.0)
    losses = [item[1] for item in outputs]
    assert all(math.isfinite(value) for value in losses)
    assert statistics.median(losses[-5:]) < 0.5 * statistics.median(losses[:5])


def test_the_weights_stay_uniform_during_the_warm_up():
    torch.manual_seed(0)
    _, rows = wf.synthetic_declines("weibull", wells=2, steps=16, seed=1)
    outputs = asyncio.run(train_outputs(rows, batch_size=2, epochs=30, report_every=1, d_model=16, num_heads=2))
    warmup = int(30 * wf.WARMUP_SHARE)
    for _, _, _, *weights in outputs[: warmup - 1]:
        assert weights == pytest.approx([0.2] * 5)


def test_a_random_model_is_drawn_from_the_seed():
    names = {wf.synthetic_declines(None, wells=1, steps=4, seed=seed)[0] for seed in range(40)}
    assert names == set(wf.DECLINE_MODELS)
    assert wf.synthetic_declines(None, seed=3) == wf.synthetic_declines(None, seed=3)
