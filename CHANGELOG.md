# Changelog

## 0.3.0

`train` now identifies the decline model behind the rates. **Its output items change**, so callers that unpack `(epoch, loss)` must read the new items; its inputs and parameters are unchanged.

### Changed

- **Outputs:** `(epoch, loss, model, w_exponential, w_harmonic, w_hyperbolic, w_weibull, w_power_law_exponential)`. `model` names the model with the largest weight; the weights are a softmax over five learnt logits and sum to 1.
- **The residual weights** are one learnt distribution per run instead of a non-negative weight per time step. They stay uniform for the first third of the epochs, then move towards the model whose closed-form curve fits the rates best; on generated data they end above 0.98 for the generating model.
- **The decline parameters** are fitted per well and per model, inside bounds that keep the families apart, instead of being predicted per time step and shared by the models.
- **The fifth model** is the power-law exponential, `q = qi exp(-Dinf t - Di t^n)`. The former form rose before it fell, and its own curve could not satisfy its residual.
- Times and rates are scaled inside `train` (rates per well), so any units work. The network no longer predicts per-step parameters; a rate is a function of its time.

### Added

- `synthetic_declines(model, wells, steps, noise, seed)`: analytic ground truth from one decline model, named or drawn at random, with the generating parameters in the five parameter columns.
- The page generates such declines, names the true model, and charts the five weights as they learn next to the loss.
- Tests: the five residuals tell their curves apart, and training on any one model's data ends with that model's weight above 0.9.

## 0.2.0

The physics now takes part in training. The operation `train` keeps its name, its parameters and its stream items, so existing calls and pages work unchanged; the losses it reports differ from 0.1.x.

### Fixed

- **The physics term used dq/dt = 0.** The residuals never saw the slope of the predicted decline, so the physics loss pulled the rate towards zero. dq/dt is now the exact slope at the sample times, taken in one backward pass.
- **The rate was fitted to all six target columns.** The predicted rate was broadcast against `q, Di, b, Dinf, n, T` and fitted to their mean. It is now compared with `q` alone, and the loss refuses tensors of different shapes.
- **The harmonic and hyperbolic residuals** used the wrong power of `(1 + b Di t)`; both now vanish on their own decline curves.
- **Declines starting at t = 0** made the Weibull residual infinite and the loss NaN from the second epoch on.
- `train` refuses an `output_dim` other than 6 or a `num_pde_constraints` other than 5 with a message, instead of failing later or broadcasting.

### Added

- `perd.toml`: the `small` compute class (2 vCPU, 8 GiB) and the Store listing.
- An icon (`favicon.ico`), drawn from the SVG sources in `assets/`.
- Tests of the build's checks, the physics and a training run, run by GitHub Actions.

### Changed

- An epoch costs one more encoder pass and one more backward pass: about 0.044 s instead of 0.029 s at the page's defaults on two CPU threads.
- The Flask-era `.dockerignore` and the VS Code settings are gone.

## 0.1.x

- The port from the first SDK to `perd_worker`: `train` as a bidirectional stream of `(epoch, loss)`, CPU-only torch, and the first self-service publish on PERD.
- The app page: the training parameters, a synthetic or uploaded decline, and the loss live on a chart.
