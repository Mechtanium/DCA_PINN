"""The five residuals: a decline model's curve zeroes its own residual and no other.

A residual is dq/dt minus the closed-form slope of a model. Along model k's
curve, residual k vanishes with model k's parameters. Any other model j,
with any parameters inside FIT_RANGES, leaves a residual: the ranges keep
the families apart (hyperbolic b clear of 0 and 1, the power-law
exponential's Di and Dinf clear of 0), so the weights have one model to find.
"""

import pytest
import torch

import workflow as wf

S = 32
#: Smallest relative residual error another model may reach on a curve: a
#: tenth of the smallest gap this search finds (2.3e-4, power-law exponential
#: data against the Weibull model).
GAP = 2e-5


def generated(model):
    """Times (S, 1), the generating parameters, in train's units (t up to 1, rates / qi)."""
    _, rows = wf.synthetic_declines(model, wells=1, steps=S, seed=0)
    first = rows[0]
    p = dict(zip(["Di", "b", "Dinf", "n", "T"], first[2:]))
    t = torch.tensor([row[0] for row in rows], dtype=torch.float64).unsqueeze(-1)
    return t, {key: torch.tensor(value, dtype=torch.float64) for key, value in {**p, "qi": 1.0}.items()}


@pytest.mark.parametrize("model", range(5), ids=wf.DECLINE_MODELS)
def test_the_slope_formula_is_the_derivative_of_the_curve(model):
    t, p = generated(model)
    t.requires_grad_(True)
    q = wf.decline_rate(model, t, p)
    torch.testing.assert_close(wf.time_derivative(q, t), wf.decline_slope(model, t, p).detach(), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("model", range(5), ids=wf.DECLINE_MODELS)
def test_a_curve_zeroes_its_own_residual(model):
    t, p = generated(model)
    t.requires_grad_(True)
    q = wf.decline_rate(model, t, p)
    dq_dt = wf.time_derivative(q, t)
    params = [p] * 5  # the generating values in all slots; only slot `model` matters here
    residual = wf.PDEConstraints()(dq_dt, t.detach(), params)[..., model]
    assert residual.abs().max() < 1e-9 * dq_dt.abs().max()


def best_fits(t, dq_dt, starts=32, steps=400):
    """The smallest relative residual errors of the five models on the slope dq_dt, over FIT_RANGES."""
    torch.manual_seed(0)
    fits = wf.DeclineParameters(num_wells=starts).double()
    with torch.no_grad():
        fits.raw.normal_(0.0, 2.0)  # spread the starts across the ranges
    optimizer = torch.optim.Adam(fits.parameters(), lr=0.05)
    scale = dq_dt.pow(2).mean()
    for _ in range(steps):
        params = fits()  # (starts,) per parameter, broadcast against t (S, 1)
        errors = torch.stack([(dq_dt - wf.decline_slope(k, t, params[k])).pow(2).mean(dim=0) for k in range(5)]) / scale
        optimizer.zero_grad()
        errors.sum().backward()
        optimizer.step()
    return errors.detach().min(dim=1).values  # (5,)


@pytest.mark.parametrize("model", range(5), ids=wf.DECLINE_MODELS)
def test_no_other_model_zeroes_the_residual_of_a_curve(model):
    t, p = generated(model)
    dq_dt = wf.decline_slope(model, t, p)
    errors = best_fits(t, dq_dt)
    others = [errors[j].item() for j in range(5) if j != model]
    assert errors[model] < 1e-5, "the generating model must fit its own curve"
    assert min(others) > GAP, dict(zip([m for j, m in enumerate(wf.DECLINE_MODELS) if j != model], others))
    assert min(others) > 10 * errors[model]


def test_the_slopes_are_finite_from_t_equal_zero():
    t = torch.linspace(0.0, 1.0, S, dtype=torch.float64).unsqueeze(-1)
    for model in range(5):
        _, p = generated(model)
        assert torch.isfinite(wf.decline_slope(model, t, p)).all(), wf.DECLINE_MODELS[model]
