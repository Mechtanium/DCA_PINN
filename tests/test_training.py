"""A short training run on synthetic Arps declines: the loss falls.

Four wells with hyperbolic declines q = qi (1 + b Di t)^(-1/b), sampled at
24 times from t = 0, with rates scaled to qi = 1 and a little noise. The rows
are streamed exactly as a client streams them: interleaved, so that row i is
time step i // 4 of well i % 4.
"""

import asyncio
import math
import random
import statistics

import torch

import workflow as wf

WELLS = [(1.0, 2.0, 0.5), (0.9, 3.0, 0.8), (0.8, 1.5, 0.3), (1.1, 2.5, 0.6)]  # (qi, Di, b)
STEPS = 24


def arps_rows(seed=0):
    rng = random.Random(seed)
    rows = []
    for step in range(STEPS):
        t = step / (STEPS - 1)
        for qi, Di, b in WELLS:
            q = qi * (1 + b * Di * t) ** (-1 / b) * (1 + 0.01 * rng.gauss(0, 1))
            rows.append((t, q, Di, b, 1e-3, 0.5, 1.0))
    return rows


async def train_losses(rows, **kwargs):
    async def stream():
        for row in rows:
            yield row

    return [loss async for _, loss in wf.train(stream(), **kwargs)]


def test_a_few_epochs_on_arps_declines_make_the_loss_fall():
    torch.manual_seed(0)
    losses = asyncio.run(train_losses(arps_rows(), batch_size=len(WELLS), epochs=40, report_every=1))
    assert len(losses) == 40
    assert all(math.isfinite(loss) for loss in losses)
    # Medians of five epochs: the fifth model's t^(-n) is singular at t = 0
    # and can spike an early epoch before the adaptive weights damp it.
    first, last = statistics.median(losses[:5]), statistics.median(losses[-5:])
    assert last < 0.5 * first, (first, last)
