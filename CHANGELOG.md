# Changelog

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
