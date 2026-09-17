"""The DCA_PINN app page — a perdlit page over `train`.

Launch this workflow as an app on PERD and this page is what visitors see:
choose or upload the decline data, set the training parameters, press
Train, and watch the loss curve as the job runs on the workstation.
"""

import math
import random

import perdlit as pl

from workflow import train

pl.title("DCA PINN")
pl.markdown(
    "A physics-informed Transformer for **decline-curve analysis**. It learns the "
    "rate decline `q(t)` while five classical DCA residuals (exponential, harmonic, "
    "hyperbolic, Weibull and stretched-exponential) pull it toward physics, with "
    "self-adaptive weights deciding which model governs each part of the curve."
)

# ── Data ────────────────────────────────────────────────────────────────
with pl.sidebar:
    pl.header("Data")
    source = pl.radio("Source", ["Synthetic decline", "Upload CSV"])
    if source == "Synthetic decline":
        seq_len = pl.slider("Sequence length", 20, 1000, 200, step=10)
        decline = pl.slider("Decline rate D", 0.005, 0.2, 0.02, step=0.005)
        noise = pl.slider("Noise (rate units)", 0.0, 50.0, 5.0, step=1.0)
        seed = pl.number_input("Random seed", 0, 9999, 7)
        uploaded = None
    else:
        uploaded = pl.file_uploader("Rows: t, q, Di, b, Dinf, n, T", type=["csv"])
        seq_len, decline, noise, seed = 0, 0.02, 0.0, 0

    pl.header("Training")
    batch_size = pl.selectbox("Batch size", [2, 4, 8], index=1)
    epochs = pl.number_input("Epochs", 1, 5000, 100)
    learning_rate = pl.number_input("Learning rate", 1e-5, 1.0, 0.001, step=0.0005)
    report_every = pl.number_input("Report every N epochs", 1, 500, 10)
    with pl.expander("Model"):
        d_model = pl.selectbox("d_model", [32, 64, 128], index=1)
        num_layers = pl.slider("Encoder layers", 1, 6, 2)
        num_heads = pl.selectbox("Attention heads", [2, 4, 8], index=1)
        lambda_pde = pl.slider("Physics weight λ_pde", 0.0, 5.0, 1.0, step=0.1)


def synthetic_rows(n: int, D: float, sigma: float, seed: int):
    """One (t, q, Di, b, Dinf, n, T) row per time step: q = qi·e^(-D·t) + noise."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n * batch_size):
        t = rng.random()
        qi = 2500 + rng.random() * 1000
        q = qi * math.exp(-D * t) + rng.random() * sigma
        rows.append((t, q, D, 0.5, 1e-3, 0.5, 1.0))
    return rows


def csv_rows(text: str):
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue
        try:
            rows.append(tuple(float(p) for p in parts))
        except ValueError:
            continue  # header or junk line
    return rows


if uploaded is not None:
    rows = csv_rows(uploaded.read().decode("utf-8", errors="replace"))
else:
    rows = synthetic_rows(seq_len, decline, noise, seed)

# ── Preview ────────────────────────────────────────────────────────────
pl.subheader("Data")
left, right = pl.columns([2, 1])
with left:
    pl.line_chart(
        sorted(rows, key=lambda r: r[0])[:400],
        x=0,
        y=1,
        labels=["t", "q"],
        height=240,
    )
with right:
    pl.metric("Rows", len(rows))
    pl.metric("Batches", len(rows) // batch_size if batch_size else 0)
    pl.caption("Each row is (t, q, Di, b, Dinf, n, T); rows are folded into sequences of `batch size`.")

with pl.expander("First rows"):
    pl.dataframe(
        [dict(zip(["t", "q", "Di", "b", "Dinf", "n", "T"], r)) for r in rows[:20]],
        height=260,
    )

# ── Train ──────────────────────────────────────────────────────────────
pl.subheader("Training")
run = pl.button(
    "Train",
    callable=train,
    args=[rows, batch_size, epochs],
    kwargs={
        "learning_rate": learning_rate,
        "report_every": report_every,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "lambda_pde": lambda_pde,
    },
    help="Runs `train` as a job on this workstation; outputs stream in below.",
)

m1, m2, m3 = pl.columns(3)
with m1:
    pl.metric("Latest loss", run, field=1, fmt=".4f")
with m2:
    pl.metric("Epoch", run, field=0)
with m3:
    pl.progress(run, total=epochs, field=0)
pl.write(run)
pl.line_chart(run, x=0, y=1, labels=["epoch", "loss"], height=320)
