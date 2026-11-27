"""dq/dt in the physics residuals: non-zero, and equal to finite differences.

Before the fix the residuals took dq/dt as zero everywhere: the times were
built without requires_grad, and the per-element loop differentiated with
respect to fresh slices of them that q never depended on.

The checks run in float64, where central differences with a small step are
accurate to about 1e-9 and a wrong derivative cannot hide in rounding.
"""

import pytest
import torch

import workflow as wf

S, B = 12, 3


def arps(t, qi, Di, b):
    """Arps hyperbolic decline q(t) = qi (1 + b Di t)^(-1/b)."""
    return qi * (1 + b * Di * t) ** (-1 / b)


def times():
    """(S, B, 1) times from t = 0, one sequence per column."""
    return torch.linspace(0.0, 2.0, S, dtype=torch.float64).repeat_interleave(B).reshape(S, B, 1)


def small_model():
    torch.manual_seed(0)
    model = wf.PINNTransformer(1, 6, 2, 16, 2, 32, 5, dropout=0.1).double()
    return model.eval()


def test_dq_dt_on_a_known_curve_matches_the_analytic_slope_and_finite_differences():
    qi, Di, b = 1000.0, 1.5, 0.6
    t = times().requires_grad_(True)
    q = arps(t, qi, Di, b)[..., 0]

    dq_dt = wf.time_derivative(q, t)

    analytic = -Di * qi * (1 + b * Di * t.detach()[..., 0]) ** (-1 / b - 1)
    h = 1e-6
    fd = (arps(t.detach() + h, qi, Di, b) - arps(t.detach() - h, qi, Di, b))[..., 0] / (2 * h)
    assert dq_dt.shape == (S, B)
    assert (dq_dt < 0).all()
    torch.testing.assert_close(dq_dt, analytic, rtol=1e-12, atol=1e-9)
    torch.testing.assert_close(dq_dt, fd, rtol=1e-7, atol=1e-6)


def test_dq_dt_of_the_model_matches_finite_differences_position_by_position():
    model = small_model()
    X = times()
    t_query = X.clone().requires_grad_(True)
    q = model(t_query, context=X)[0][..., 0]

    dq_dt = wf.time_derivative(q, t_query).detach()

    h = 1e-6
    fd = torch.zeros(S, B, dtype=torch.float64)
    with torch.no_grad():
        for s in range(S):
            for col in range(B):
                plus, minus = X.clone(), X.clone()
                plus[s, col, 0] += h
                minus[s, col, 0] -= h
                q_plus = model(plus, context=X)[0][s, col, 0]
                q_minus = model(minus, context=X)[0][s, col, 0]
                fd[s, col] = (q_plus - q_minus) / (2 * h)
    assert dq_dt.abs().min() > 1e-6, "dq/dt must not vanish"
    torch.testing.assert_close(dq_dt, fd, rtol=1e-6, atol=1e-8)


def test_a_query_time_moves_only_its_own_prediction():
    model = small_model()
    X = times()
    moved = X.clone()
    moved[4, 1, 0] += 0.3
    with torch.no_grad():
        before = model(X, context=X)[0]
        after = model(moved, context=X)[0]
    changed = (after - before).abs().amax(dim=-1) > 0
    assert changed[4, 1]
    changed[4, 1] = False
    assert not changed.any()


def test_reading_the_times_as_queries_and_context_changes_no_prediction():
    model = small_model()
    X = times()
    with torch.no_grad():
        plain_y, plain_w = model(X)
        query_y, query_w = model(X, context=X)
    torch.testing.assert_close(query_y, plain_y, rtol=0, atol=1e-12)
    torch.testing.assert_close(query_w, plain_w, rtol=0, atol=1e-12)


def test_the_exponential_residual_vanishes_on_an_exponential_decline():
    qi, Di = 1000.0, 1.5
    t = times().requires_grad_(True)
    q = qi * torch.exp(-Di * t)
    params = torch.tensor([Di, 0.5, 1e-3, 0.5, 1.0], dtype=torch.float64).expand(S, B, 5)
    residuals = wf.PDEConstraints(5)(torch.cat([q, params], dim=-1), t)
    # With dq/dt taken as zero this residual would be Di * q, up to 1500.
    torch.testing.assert_close(residuals[..., 0], torch.zeros(S, B, dtype=torch.float64), rtol=0, atol=1e-9)


def test_times_without_grad_are_refused_instead_of_giving_zero():
    t = times()
    q = arps(t, 1000.0, 1.5, 0.6)[..., 0]
    with pytest.raises(ValueError, match="requires_grad"):
        wf.time_derivative(q, t)
