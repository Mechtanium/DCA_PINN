"""The DCA_PINN app page: a perdlit page over `train`.

Launch this workflow as an app on PERD and visitors see this page: generate
declines from a decline model (or upload rates), set the training
parameters, press Train, and watch the network's weights settle on the
model behind the data.
"""

import perdlit as pl

from workflow import DECLINE_MODELS, synthetic_declines, train

NAMES = {
    "exponential": "Exponential",
    "harmonic": "Harmonic",
    "hyperbolic": "Hyperbolic",
    "weibull": "Weibull",
    "power_law_exponential": "Power-law exponential",
}
COLUMNS = ["t", "q", "Di", "b", "Dinf", "n", "T"]

pl.title("DCA PINN")
pl.markdown(
    "A physics-informed Transformer for **decline-curve analysis**. It learns the rate "
    "decline `q(t)` of a few wells and identifies the decline model behind it: five "
    "residuals (exponential, harmonic, hyperbolic, Weibull and power-law exponential) "
    "carry learnt weights, and training drives the weight of the generating model to 1."
)

# ── Data ────────────────────────────────────────────────────────────────
with pl.sidebar:
    pl.header("Data")
    source = pl.radio("Source", ["Synthetic decline", "Upload CSV"])
    if source == "Synthetic decline":
        choice = pl.selectbox("Decline model", ["Random"] + [NAMES[m] for m in DECLINE_MODELS])
        wells = pl.selectbox("Wells", [1, 2, 4, 8], index=2)
        steps = pl.slider("Time steps per well", 20, 500, 50, step=10)
        noise = pl.slider("Noise (% of the rate)", 0.0, 5.0, 0.0, step=0.5)
        seed = pl.number_input("Random seed", 0, 9999, 7)
        uploaded = None
    else:
        uploaded = pl.file_uploader("Rows: t, q, Di, b, Dinf, n, T", type=["csv"])
        wells = pl.selectbox("Wells in the file (rows interleaved by step)", [1, 2, 4, 8], index=2)

    pl.header("Training")
    epochs = pl.number_input("Epochs", 1, 5000, 100)
    learning_rate = pl.number_input("Learning rate", 1e-5, 1.0, 0.001, step=0.0005)
    report_every = pl.number_input("Report every N epochs", 1, 500, 5)
    with pl.expander("Model"):
        d_model = pl.selectbox("d_model", [32, 64], index=1)
        num_layers = pl.slider("Encoder layers", 1, 4, 2)
        num_heads = pl.selectbox("Attention heads", [2, 4, 8], index=1)
        lambda_pde = pl.slider("Physics weight λ_pde", 0.0, 5.0, 1.0, step=0.1)


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
    true_model = None
    rows = csv_rows(uploaded.read().decode("utf-8", errors="replace"))
elif source == "Synthetic decline":
    model = None if choice == "Random" else DECLINE_MODELS[[NAMES[m] for m in DECLINE_MODELS].index(choice)]
    true_model, rows = synthetic_declines(model, wells=wells, steps=steps, noise=noise / 100, seed=seed)
else:
    true_model, rows = None, []

# ── Preview ────────────────────────────────────────────────────────────
pl.subheader("Data")
left, right = pl.columns([2, 1])
with left:
    # One line per well: row i is step i // wells of well i % wells.
    by_step = [rows[i : i + wells] for i in range(0, len(rows) - wells + 1, wells)]
    table = [{"t": step[0][0], **{f"well {w + 1}": row[1] for w, row in enumerate(step)}} for step in by_step]
    pl.line_chart(
        table,
        x="t",
        y=[f"well {w + 1}" for w in range(wells)],
        labels=["t", "q"],
        legend=[f"well {w + 1}" for w in range(wells)],
        height=260,
        key="data",
    )
with right:
    pl.metric("True model", NAMES[true_model] if true_model else "unknown")
    pl.metric("Wells", wells)
    pl.metric("Time steps", len(rows) // wells if wells else 0)
    pl.caption("Rows are (t, q, Di, b, Dinf, n, T), interleaved by step; train reads t and q.")

with pl.expander("First rows"):
    pl.dataframe([dict(zip(COLUMNS, r)) for r in rows[:20]], height=260)

# ── Train ──────────────────────────────────────────────────────────────
pl.subheader("Training")
run = pl.button(
    "Train",
    callable=train,
    args=[rows, wells, epochs],
    kwargs={
        "learning_rate": learning_rate,
        "report_every": report_every,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "lambda_pde": lambda_pde,
    },
    help="Runs `train` as a job on this workspace; outputs stream in below.",
)

m1, m2, m3, m4 = pl.columns(4)
with m1:
    pl.metric("Identified model", run, field=2)
with m2:
    pl.metric("True model", NAMES[true_model] if true_model else "unknown")
with m3:
    if true_model:
        pl.metric("Weight of the true model", run, field=3 + DECLINE_MODELS.index(true_model), fmt=".3f")
    else:
        pl.metric("Latest loss", run, field=1, fmt=".4g")
with m4:
    pl.progress(run, total=epochs, field=0)
pl.write(run)

pl.line_chart(
    run,
    x=0,
    y=[3, 4, 5, 6, 7],
    labels=["epoch", "weight"],
    legend=[NAMES[m] for m in DECLINE_MODELS],
    height=300,
)
pl.caption(
    "The weights stay uniform for the first third of the epochs while the five models fit "
    "their parameters, then move towards the model whose curve fits the rates best."
)
pl.line_chart(run, x=0, y=1, labels=["epoch", "loss"], y_scale="log", height=260)
