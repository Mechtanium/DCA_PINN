"""The five decline residuals vanish on their own decline curves.

A residual is dq/dt plus the closed-form decline rate of its model, so it is
zero along that model's q(t). The harmonic and hyperbolic forms used the
wrong power of (1 + b Di t), and the Weibull form was infinite at t = 0.
"""

import math

import pytest
import torch

import workflow as wf

S, B = 16, 2
QI, DI, BEXP, DINF, N, T_SCALE = 1000.0, 1.5, 0.6, 0.05, 0.7, 2.0


def curve(model, t):
    """q(t) of decline model ``model`` (0-4) with the parameters above."""
    if model == 0:
        return QI * torch.exp(-DI * t)
    if model == 1:
        return QI / (1 + DI * t)
    if model == 2:
        return QI * (1 + BEXP * DI * t) ** (-1 / BEXP)
    if model == 3:
        return QI * torch.exp(-((t / T_SCALE) ** N))
    # Model 5's qi is the rate scale at t = 1: start that sequence at t = 1, without Dinf.
    return QI * torch.exp((DI / (1 - N)) * (t ** (1 - N) - 1))


def residuals_along(model, t0):
    t = (t0 + torch.linspace(0.0, 3.0, S, dtype=torch.float64)).repeat_interleave(B).reshape(S, B, 1)
    t.requires_grad_(True)
    q = curve(model, t)
    dinf = 0.0 if model == 4 else DINF
    params = torch.tensor([DI, BEXP, dinf, N, T_SCALE], dtype=torch.float64).expand(S, B, 5)
    return wf.PDEConstraints(5)(torch.cat([q, params], dim=-1), t)


@pytest.mark.parametrize(
    "model, name, t0",
    [(0, "exponential", 0.0), (1, "harmonic", 0.0), (2, "hyperbolic", 0.0), (3, "Weibull", 0.0), (4, "power-law", 1.0)],
)
def test_each_residual_vanishes_on_its_own_curve(model, name, t0):
    residual = residuals_along(model, t0)[..., model]
    if name == "Weibull":
        # For n < 1 the Weibull curve itself falls infinitely fast at t = 0.
        residual = residual[1:]
    torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=1e-8, msg=name)


@pytest.mark.parametrize("model", [0, 1, 2])
def test_all_five_residuals_are_finite_from_t_equal_zero(model):
    # A smooth decline from t = 0, the usual first step of a production
    # history: the Weibull residual's (t/T)^(n-1) used to be infinite there,
    # which made the loss infinite and its gradients NaN.
    assert torch.isfinite(residuals_along(model, 0.0)).all()


def test_the_harmonic_and_hyperbolic_residuals_tell_their_curves_apart():
    # On a hyperbolic decline the hyperbolic residual vanishes and the harmonic
    # one does not; the self-adaptive weights rely on that contrast.
    residuals = residuals_along(2, 0.0)
    assert residuals[1:, :, 2].abs().max() < 1e-8
    assert residuals[1:, :, 1].abs().max() > 0.01 * QI * DI / math.e
