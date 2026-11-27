<img src="assets/favicon.svg" width="88" height="88" alt="" align="right">

# DCA PINN — a PERD workflow

Decline curve analysis (DCA) is one of the primary ways to estimate expected rate declines and total recovery in pressurised porous-media flow systems. This workflow trains a physics-informed neural network with a Transformer core on the rates of a few wells and identifies the decline model behind them. Five classical models (exponential, harmonic, hyperbolic, Weibull and power-law exponential) act as physics residuals with learnt weights, and training drives the weight of the generating model to 1 and the others to 0.

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
python -m perd workflow publish --git https://github.com/Mechtanium/DCA_PINN --name "DCA PINN" --version 0.3.0
```

PERD builds the image from its own Dockerfile (`python:3.12-slim`, `requirements.txt`, then `perd-worker`), reads `perd.toml`, describes the contract, dry-runs the page and lists the version. API keys are issued under **Settings › Account › API keys** in the console.

## Using it from Python

```python
import asyncio
import os
import random

from perd import Workstation


def arps_rows(wells=4, steps=50, Di=8.0, b=0.5):
    """Hyperbolic declines sampled after t = 0, interleaved by step."""
    for step in range(1, steps + 1):
        t = step / steps
        for _ in range(wells):
            q = 2000 * (1 + b * Di * t) ** (-1 / b) * (1 + 0.005 * random.gauss(0, 1))
            yield (t, q, Di, b, 0.0, 0.0, 0.0)


async def main():
    ws = await Workstation.connect(api_key=os.environ["PERD_API_KEY"], workflows=["dca_pinn"])
    try:
        async for epoch, loss, model, *weights in ws.workflows.dca_pinn.train(arps_rows(), batch_size=4, epochs=100):
            print(f"epoch {epoch}: loss {loss:.4f}, model {model}, weights {[round(w, 3) for w in weights]}")
    finally:
        await ws.delete()


asyncio.run(main())
```

`Workstation.connect` launches a workspace running this workflow and returns once it is ready; `help(ws.workflows.dca_pinn.train)` lists the parameters. With a clone of this repository, `workflow.synthetic_declines(model, wells, steps, noise, seed)` generates the same kind of rows from any of the five models, or from one drawn at random.

### The input stream

`train` reads one row per time step, `(t, q, Di, b, Dinf, n, T)`, and interleaves the rows into `batch_size` wells: row `i` is step `i // batch_size` of well `i % batch_size`. A batch of four wells is therefore streamed step by step, the four wells' rows for a step one after the other.

- Times are scaled by the longest time and rates by their well's largest rate, so any units work.
- `Di, b, Dinf, n, T` hold the generating parameters (`synthetic_declines` writes them, 0 for the parameters a model does not use). `train` does not read them: it fits the parameters of all five models itself.
- Residuals skip `t <= 0`, where the Weibull and power-law exponential curves fall infinitely fast.

### The output stream

On the first epoch and then once per `report_every` epochs, `train` streams `(epoch, loss, model, w_exponential, w_harmonic, w_hyperbolic, w_weibull, w_power_law_exponential)`: `model` is the name of the model with the largest weight, and the five weights sum to 1. They stay at 0.2 for the first third of the epochs.

## The app page

[`page.py`](page.py) gives the workflow a face. It is a **perdlit** page, the Streamlit-shaped UI library inside `perd-worker`, and it runs on the workspace's own compute: on a workspace's desktop, the workflow's icon opens it in a tab. The page generates declines with `synthetic_declines` (a model or Random, wells, time steps, noise and seed) or reads an uploaded CSV of `t, q, Di, b, Dinf, n, T` rows, shows the wells' curves and the true model, and puts `train` on a button:

```python
run = pl.button("Train", callable=train, args=[rows, wells, epochs], kwargs={"learning_rate": learning_rate, ...})
pl.metric("Identified model", run, field=2)
pl.line_chart(run, x=0, y=[3, 4, 5, 6, 7], legend=["Exponential", "Harmonic", "Hyperbolic", "Weibull", "Power-law exponential"])
pl.line_chart(run, x=0, y=1, labels=["epoch", "loss"], y_scale="log")
```

Pressing **Train** runs `train` as a job on the workspace's orchestrator, the same durable path a Python caller takes. Its items stream into the identified model, the true model's weight, the progress bar and the charts of the five weights and the loss, for all viewers of the page.

## Testing

```bash
python -m venv .venv && source .venv/bin/activate
pip install "perd-worker[page]" pytest -r requirements.txt
pytest
```

The tests run the checks of PERD's build (`perd.toml`, the contract, the page, the icon), test the physics (dq/dt against finite differences, the five residuals on their own curves, the shapes the loss compares) and check that training on any one model's declines drives that model's weight above 0.9. GitHub Actions runs them on pushes and pull requests.

To serve the worker locally:

```bash
python -m perd_worker.serve --module workflow --name dca_pinn --port 50051
```

## Model

`PINNTransformer` is the rate network. It embeds the time, adds a learnt code per well, and reads the times twice: as a context stream of ordinary self-attention, and as a query stream whose positions attend only to the context and carry no positional encoding. A rate is then a function of its time, and one backward pass gives its slope at all times (`time_derivative`).

`PDEConstraints` evaluates five residuals, dq/dt minus the closed-form slope of a model:

| Model | q(t) | Residual |
| --- | --- | --- |
| Exponential | `qi exp(-Di t)` | `dq/dt + Di qi exp(-Di t)` |
| Harmonic | `qi / (1 + Di t)` | `dq/dt + Di qi / (1 + Di t)^2` |
| Hyperbolic | `qi (1 + b Di t)^(-1/b)` | `dq/dt + Di qi / (1 + b Di t)^(1/b + 1)` |
| Weibull | `qi exp(-(t/T)^n)` | `dq/dt + (n qi / T) (t/T)^(n-1) exp(-(t/T)^n)` |
| Power-law exponential | `qi exp(-Dinf t - Di t^n)` | `dq/dt + (Dinf + n Di t^(n-1)) qi exp(-Dinf t - Di t^n)` |

An epoch has three parts:

1. **The decline parameters.** `DeclineParameters` holds the parameters of all five models for all wells (`qi` and the model's own constants, constant over a well's history, inside `FIT_RANGES`). Five Adam steps fit them to the rates through the closed-form curves, all five models unweighted, so a model the weights have not chosen yet still reaches its best fit.
2. **The network.** `PINNLoss` adds the rates' mean squared error to the residual errors of the fitted models, weighted by the current weights.
3. **The weights.** `ModelSelector` holds one logit per model; the weights are their softmax, one distribution per run. After the warm-up the logits learn from the logarithms of the models' misfits, so the model whose curve fits the rates wins by ratio, at any scale of the errors.

The bounds in `FIT_RANGES` keep the families apart: hyperbolic `b` stays clear of 0 (exponential) and 1 (harmonic), and the power-law exponential's `Di` and `Dinf` clear of 0 (Weibull and exponential). A model's curve therefore zeroes its own residual and no other (`tests/test_residuals.py`). The network's own slope is too rough after a few hundred epochs to choose between families a few percent apart, so the weights learn from the closed-form fits, and the physics term then pulls the network towards the chosen model.

## License

See [LICENSE](LICENSE).
