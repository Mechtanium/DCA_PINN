# DCA_PINN — a PERD workflow

Decline Curve Analysis (DCA) is one of the primary ways to estimate expected rate declines and total recovery in pressurised porous-media flow systems. This workflow trains a Physics-Informed Neural Network with a Transformer core to predict rate decline while learning, through self-adaptive weights, which of five classical DCA models (exponential, harmonic, hyperbolic, Weibull-type, and a stretched-exponential form) governs each part of each sequence.

It is also the **template for writing your own PERD workflow**: one Python module, one decorator, no Dockerfile.

## How a workflow is written

```python
from perd_worker import workflow, WorkflowStreamInput, WorkflowStreamOutput

@workflow.bi_di
async def train(
    inputStream: WorkflowStreamInput[float, float, float, float, float, float, float],
    epochs: int = 100,
    learning_rate: float = 0.001,
) -> WorkflowStreamOutput[int, float]:
    async for t, q, Di, b, Dinf, n, T in inputStream:
        ...
    yield epoch, loss
```

- Decorate with `@workflow.unary`, `.input_stream`, `.output_stream`, or `.bi_di`; the decorated functions are the workflow's public interface.
- Parameter and stream item types may be `float`, `int`, `str`, `bool`, or `bytes`. Multi-field stream items are tuples.
- The module must be importable from the repository root and expose the module-level `workflow` registry; this repository's module is `workflow.py`.
- `requirements.txt` is installed into the worker image before the SDK.

## Publishing it on PERD

Sign in on the PERD website, open **Store → Publish from GitHub**, and paste this repository's URL. PERD clones it, builds the worker image, describes the contract, and lists the workflow under the id `dca_pinn` (the snake_case slug of the name you give it). The same thing from a terminal:

```bash
pip install perd
export PERD_API_KEY=pak_...
python -m perd workflow publish --git https://github.com/Mechtanium/DCA_PINN --name "DCA PINN"
```

## Using it from Python

```python
import asyncio
import math
import random

from perd import Workstation

API_KEY = "pak_..."          # Store → API Keys on the website

def samples(seq_len=200, batch_size=4, D=2e-2):
    """One (t, q, Di, b, Dinf, n, T) row per time step: t in [0, 1], q = qi·e^(-D·t) + noise."""
    for _ in range(seq_len * batch_size):
        t = random.random()
        qi = 2500 + random.random() * 1000
        q = qi * math.exp(-D * t) + random.random() * 5
        yield (t, q, D, 0.5, 1e-3, 0.5, 1.0)

async def main():
    ws = await Workstation.connect(api_key=API_KEY, workflows=["dca_pinn"])
    async for epoch, loss in ws.workflows.dca_pinn.train(samples(), epochs=50, learning_rate=0.001):
        print(f"epoch {epoch}: loss {loss:.4f}")

asyncio.run(main())
```

`Workstation.connect` finds the control plane on its own (or uses `PERD_API_URL` if you set one), resolves the workflow's contract, provisions a workstation running this image, and returns a handle whose methods are materialised from the contract — `help(ws.workflows.dca_pinn.train)` shows the parameters above.

## The app page

[`page.py`](page.py) gives the workflow a face. It is a **perdlit** page — the Streamlit-shaped UI library that ships inside `perd-worker` — and it runs on the workstation itself when the workflow is launched as an app from the PERD store (**Launch app** on the store entry). The page collects the data (a synthetic decline with sequence length, decline rate, noise and seed, or an uploaded CSV of `t, q, Di, b, Dinf, n, T` rows) and the training parameters, previews the rows, and puts `train` on a button:

```python
run = pl.button("Train", callable=train, args=[rows, batch_size, epochs], kwargs={"learning_rate": learning_rate, ...})
pl.metric("Latest loss", run, field=1, fmt=".4f")
pl.progress(run, total=epochs, field=0)
pl.line_chart(run, x=0, y=1, labels=["epoch", "loss"])
```

Pressing **Train** runs `train` as a job on the workstation's orchestrator — the same durable path a Python caller takes — and every `(epoch, loss)` it yields streams into the metric, the progress bar and the chart for everyone looking at the page. The app lives at `https://perd.app/app/<slug>/`; who can open it (anyone with the link, signed-in users, or an invite list) and what they may do (view, run, or edit and run) is chosen at launch and can be changed afterwards.

To check the page renders before publishing:

```bash
pip install -r requirements.txt 'perd-worker[page]'
python -c "from perdlit._script import check_page; print(check_page('page', 'workflow'))"
```

## Running the module locally

```bash
pip install -r requirements.txt perd-worker
python -m perd_worker.serve --module workflow --name dca_pinn --port 50051
```

## Model

`PINNTransformer` embeds the time input, runs it through a stack of Transformer encoder layers, and predicts six outputs per step — `q` and the DCA parameters `Di, b, Dinf, n, T` — plus five self-adaptive non-negative weights. `PDEConstraints` evaluates the five decline-model residuals and `PINNLoss` combines the data MSE with the weighted physics residuals.

## License

See [LICENSE](LICENSE).
