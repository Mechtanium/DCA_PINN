<img src="assets/favicon.svg" width="88" height="88" alt="" align="right">

# DCA PINN — a PERD workflow

Decline curve analysis (DCA) is one of the primary ways to estimate expected rate declines and total recovery in pressurised porous-media flow systems. This workflow trains a physics-informed neural network with a Transformer core on decline data: it fits the observed rate while five classical decline models (exponential, harmonic, hyperbolic, Weibull and a power-law form) act as physics residuals, and self-adaptive weights learn the model in charge of the individual parts of a sequence.

On PERD it is the workflow `dca_pinn`: a Python API, and an app page that trains it from the browser.

## Files

| File | Purpose |
| --- | --- |
| `workflow.py` | The model and `train`, the operation PERD serves. |
| `page.py` | The app page, written with perdlit. |
| `perd.toml` | The compute class (`small`) and the Store listing. |
| `requirements.txt` | CPU-only torch and numpy. The build installs `perd-worker` itself. |
| `favicon.ico` | The icon on the desktop and in the Store; `assets/` holds its SVG sources and `build_favicon.py`. |
| `CHANGELOG.md` | Shown on the workflow's Changelog tab. |
| `tests/` | The build's checks, the physics, and a short training run. |

## Publishing it on PERD

In the PERD console, open your project, then **Workflows › Publish new**, paste this repository's address and name the workflow `DCA PINN` (its id is the slug `dca_pinn`). Later versions go through **Publish new version** on the workflow's page. The same from a terminal:

```bash
pip install perd
export PERD_API_KEY=pak_...
python -m perd workflow publish --git https://github.com/Mechtanium/DCA_PINN --name "DCA PINN" --version 0.2.0
```

PERD builds the image from its own Dockerfile (`python:3.12-slim`, `requirements.txt`, then `perd-worker`), reads `perd.toml`, describes the contract, dry-runs the page and lists the version. API keys are issued under **Settings › Account › API keys** in the console.

## Using it from Python

```python
import asyncio
import os
import random

from perd import Workstation


def arps_rows(wells=4, steps=200, Di=2.0, b=0.5):
    """Hyperbolic declines from t = 0, rates scaled to qi = 1, interleaved by step."""
    for step in range(steps):
        t = step / (steps - 1)
        for _ in range(wells):
            q = (1 + b * Di * t) ** (-1 / b) * (1 + 0.01 * random.gauss(0, 1))
            yield (t, q, Di, b, 1e-3, 0.5, 1.0)


async def main():
    ws = await Workstation.connect(api_key=os.environ["PERD_API_KEY"], workflows=["dca_pinn"])
    try:
        async for epoch, loss in ws.workflows.dca_pinn.train(arps_rows(), batch_size=4, epochs=50):
            print(f"epoch {epoch}: loss {loss:.4f}")
    finally:
        await ws.delete()


asyncio.run(main())
```

`Workstation.connect` launches a workspace running this workflow and returns once it is ready; `help(ws.workflows.dca_pinn.train)` lists the parameters.

### The input stream

`train` reads one row per time step, `(t, q, Di, b, Dinf, n, T)`, and interleaves the rows into `batch_size` sequences: row `i` is step `i // batch_size` of sequence `i % batch_size`. A batch of four wells is therefore streamed step by step, the four wells' rows for a step one after the other.

- Start the sequences at `t = 0`: the physics takes the first rate of a sequence as its initial rate `qi`.
- The loss fits the predicted rate to `q`. The network infers the five parameters through the physics, so the columns `Di, b, Dinf, n, T` are read but not compared with anything.
- Rates scaled to order one (divided by the initial rate) train within tens of epochs, as in `tests/test_training.py`. With raw rates in the thousands the data term outweighs the physics, and the loss moves little in a hundred epochs.

`train` streams out `(epoch, loss)` on the first epoch and then once per `report_every` epochs.

## The app page

[`page.py`](page.py) gives the workflow a face. It is a **perdlit** page, the Streamlit-shaped UI library inside `perd-worker`, and it runs on the workspace's own compute: on a workspace's desktop, the workflow's icon opens it in a tab. The page collects the data (a synthetic decline with sequence length, decline rate, noise and seed, or an uploaded CSV of `t, q, Di, b, Dinf, n, T` rows) and the training parameters, previews the rows, and puts `train` on a button:

```python
run = pl.button("Train", callable=train, args=[rows, batch_size, epochs], kwargs={"learning_rate": learning_rate, ...})
pl.metric("Latest loss", run, field=1, fmt=".4f")
pl.progress(run, total=epochs, field=0)
pl.line_chart(run, x=0, y=1, labels=["epoch", "loss"])
```

Pressing **Train** runs `train` as a job on the workspace's orchestrator, the same durable path a Python caller takes, and the `(epoch, loss)` items it yields stream into the metric, the progress bar and the chart for all viewers of the page.

## Testing

```bash
python -m venv .venv && source .venv/bin/activate
pip install "perd-worker[page]" pytest -r requirements.txt
pytest
```

The tests run the checks of PERD's build (`perd.toml`, the contract, the page, the icon), test the physics (dq/dt against finite differences, the five residuals on their own curves, the shapes the loss compares) and train for 40 epochs on synthetic Arps declines. GitHub Actions runs them on pushes and pull requests.

To serve the worker locally:

```bash
python -m perd_worker.serve --module workflow --name dca_pinn --port 50051
```

## Model

`PINNTransformer` embeds the time, runs it through a stack of Transformer encoder layers, and predicts six outputs per step (the rate `q` and the parameters `Di, b, Dinf, n, T`) plus five non-negative self-adaptive weights. `PDEConstraints` evaluates five residuals, dq/dt plus the decline rate of a model:

| Model | q(t) | Residual |
| --- | --- | --- |
| Exponential | `qi exp(-Di t)` | `dq/dt + Di qi exp(-Di t)` |
| Harmonic | `qi / (1 + Di t)` | `dq/dt + Di qi / (1 + Di t)^2` |
| Hyperbolic | `qi (1 + b Di t)^(-1/b)` | `dq/dt + Di qi / (1 + b Di t)^(1/b + 1)` |
| Weibull | `qi exp(-(t/T)^n)` | `dq/dt + (n qi / T) (t/T)^(n-1) exp(-(t/T)^n)` |
| Power law | `qi exp(Di/(1-n) (t^(1-n) - 1) - Dinf t)` | `dq/dt - qi (Di t^(-n) - Dinf) exp(Di/(1-n) (t^(1-n) - 1) - Dinf t)` |

`PINNLoss` adds the mean squared error of the rate to the weighted squared residuals, averaged over the steps.

dq/dt is the slope of the predicted decline at the sample times, with the observed times held fixed. During training the encoder reads the times twice: as a context stream of ordinary self-attention, and as a query stream whose positions attend only to the context. A query time then moves only its own prediction, so one backward pass gives the slopes at all times (`time_derivative`), and the predictions equal those of a single stream.

## License

See [LICENSE](LICENSE).
