"""DCA_PINN: a Physics-Informed Transformer for decline-curve analysis, as a
PERD workflow.

``train`` fits a Transformer to the rates of a few wells and identifies the
decline model behind them: exponential, harmonic, hyperbolic, Weibull or
power-law exponential. ``synthetic_declines`` generates analytic ground
truth from one of those models, for the page and the tests.

PERD serves the module-level ``workflow`` registry (from ``perd_worker``):
publish this repository on the website (or with ``python -m perd workflow
publish --git <this repo>``), start a workspace on it, and call
``ws.workflows.dca_pinn.train(...)`` from the ``perd`` library.
"""

import math
import random

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch_size, d_model)
        # Add positional encoding to the input. We only use up to the current sequence length.
        x = x + self.pe[:x.size(0)]
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = query.size(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.query_linear(query).view(-1, batch_size, self.num_heads, self.head_dim).transpose(0, 2)
        key = self.key_linear(key).view(-1, batch_size, self.num_heads, self.head_dim).transpose(0, 2)
        value = self.value_linear(value).view(-1, batch_size, self.num_heads, self.head_dim).transpose(0, 2)

        # 2) Apply attention on all the projected vectors in batch.
        # scores: (num_heads, batch_size, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # Mask out future positions
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        # x: (num_heads, batch_size, seq_len, head_dim)
        x = torch.matmul(p_attn, value)

        # Concat multi-head attention output
        # x: (seq_len, batch_size, d_model)
        x = x.transpose(0, 2).contiguous().view(-1, batch_size, self.d_model)

        # 3) "Concat" and apply final linear.
        return self.output_linear(x)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch_size, d_model)
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, context: torch.Tensor = None) -> torch.Tensor:
        # Self-attention block with residual connection and layer normalization.
        # With a context, the positions of x attend to the context's keys and
        # values instead of to each other (see PINNTransformer.forward).
        kv = x if context is None else context
        attn_output = self.self_attn(x, kv, kv, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward block with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, context: torch.Tensor = None) -> torch.Tensor:
        # x: (seq_len, batch_size, d_model)
        if context is not None:
            # Two streams: the context runs through the layers as ordinary
            # self-attention, and the query stream x reads the context's state
            # at the entry of the same layer. The query carries no positional
            # encoding, so its outputs are functions of its times, not of slots.
            context = self.pos_encoder(context)
            for i, layer in enumerate(self.layers):
                query = layer(x, mask, context=context)
                if i + 1 < len(self.layers):
                    context = layer(context, mask)
                x = query
            return self.norm(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x) # Final layer norm, often applied after the last encoder layer
        return x


# ── The five decline models ─────────────────────────────────────────────────
#: The decline models, in the order of their residuals and of their weights.
DECLINE_MODELS = ("exponential", "harmonic", "hyperbolic", "weibull", "power_law_exponential")

#: Bounds of the parameters train fits for all models and wells, in train's
#: normalised units (times divided by the longest time, rates by the largest
#: rate). The bounds keep the families apart: hyperbolic b stays clear of 0
#: (exponential) and 1 (harmonic), and the power-law exponential's D_inf and
#: Di clear of 0 (Weibull and exponential), so only the generating model can
#: zero its residual.
FIT_RANGES = {
    "exponential": {"qi": (0.3, 5.0), "Di": (0.05, 30.0)},
    "harmonic": {"qi": (0.3, 5.0), "Di": (0.05, 100.0)},
    "hyperbolic": {"qi": (0.3, 5.0), "Di": (0.05, 100.0), "b": (0.25, 0.75)},
    "weibull": {"qi": (0.3, 5.0), "T": (0.01, 10.0), "n": (0.15, 0.85)},
    "power_law_exponential": {"qi": (0.3, 5.0), "Di": (0.5, 10.0), "Dinf": (1.2, 10.0), "n": (0.15, 0.75)},
}
#: Parameters fitted on a linear scale; the others are rates and times, fitted
#: on a log scale.
LINEAR_PARAMETERS = {"b", "n"}

#: Parameter ranges of synthetic_declines (times from 0 to 1), inside FIT_RANGES.
SYNTHETIC_RANGES = {
    "exponential": {"Di": (2.0, 4.0)},
    "harmonic": {"Di": (5.0, 15.0)},
    "hyperbolic": {"Di": (5.0, 15.0), "b": (0.4, 0.6)},
    "weibull": {"T": (0.1, 0.5), "n": (0.3, 0.5)},
    "power_law_exponential": {"Di": (1.0, 2.0), "Dinf": (2.5, 4.0), "n": (0.3, 0.5)},
}
#: Initial rates of the synthetic wells, in rate units.
SYNTHETIC_QI = (800.0, 3000.0)


def decline_rate(model: int, t: torch.Tensor, p: dict) -> torch.Tensor:
    """q(t) of decline model ``model`` (an index into DECLINE_MODELS) with parameters ``p``."""
    qi = p["qi"]
    if model == 0:  # exponential
        return qi * torch.exp(-p["Di"] * t)
    if model == 1:  # harmonic
        return qi / (1 + p["Di"] * t)
    if model == 2:  # hyperbolic (Arps)
        return qi * (1 + p["b"] * p["Di"] * t) ** (-1 / p["b"])
    if model == 3:  # Weibull (stretched exponential)
        return qi * torch.exp(-((t / p["T"]) ** p["n"]))
    # power-law exponential (Ilk et al., 2008)
    return qi * torch.exp(-p["Dinf"] * t - p["Di"] * t ** p["n"])


def decline_slope(model: int, t: torch.Tensor, p: dict) -> torch.Tensor:
    """dq/dt of decline model ``model``: the closed form its residual compares with."""
    qi = p["qi"]
    # Powers of t with a negative exponent (Weibull, power-law exponential)
    # are infinite at t = 0; they read t from 1e-6.
    t_stable = torch.clamp(t, min=1e-6)
    if model == 0:
        return -p["Di"] * qi * torch.exp(-p["Di"] * t)
    if model == 1:
        return -p["Di"] * qi / (1 + p["Di"] * t) ** 2
    if model == 2:
        return -p["Di"] * qi * (1 + p["b"] * p["Di"] * t) ** (-1 / p["b"] - 1)
    if model == 3:
        x = t_stable / p["T"]
        return -(p["n"] / p["T"]) * x ** (p["n"] - 1) * qi * torch.exp(-(x ** p["n"]))
    return -(p["Dinf"] + p["n"] * p["Di"] * t_stable ** (p["n"] - 1)) * decline_rate(4, t_stable, p)


def synthetic_declines(model=None, wells: int = 4, steps: int = 50, noise: float = 0.0, seed: int = 0):
    """Analytic ground truth for train: ``(model name, rows)``.

    One decline model generates all the wells: ``model`` names it (a name from
    DECLINE_MODELS or its index), or ``None`` draws it at random from
    ``seed``. The wells draw their own initial rates from SYNTHETIC_QI and their
    parameters from SYNTHETIC_RANGES, and are sampled at ``steps`` times
    ``1/steps, 2/steps, ..., 1`` (after t = 0, where the Weibull and
    power-law exponential curves fall infinitely fast). ``noise`` is a
    relative Gaussian error on the rates.

    The rows are train's input, ``(t, q, Di, b, Dinf, n, T)`` interleaved by
    step (row ``i`` is step ``i // wells`` of well ``i % wells``), with the
    generating parameters in the last five columns and 0 for the parameters
    the model does not use.
    """
    rng = random.Random(seed)
    if model is None:
        index = rng.randrange(len(DECLINE_MODELS))
    elif isinstance(model, str):
        index = DECLINE_MODELS.index(model)
    else:
        index = int(model)
    name = DECLINE_MODELS[index]
    params = []
    for _ in range(int(wells)):
        p = {"qi": rng.uniform(*SYNTHETIC_QI), "Di": 0.0, "b": 0.0, "Dinf": 0.0, "n": 0.0, "T": 0.0}
        for key, (low, high) in SYNTHETIC_RANGES[name].items():
            p[key] = rng.uniform(low, high)
        params.append(p)
    times = torch.arange(1, int(steps) + 1, dtype=torch.float64) / int(steps)
    curves = [decline_rate(index, times, {k: torch.tensor(v, dtype=torch.float64) for k, v in p.items()}) for p in params]
    rows = []
    for step in range(int(steps)):
        for well, p in enumerate(params):
            q = float(curves[well][step])
            if noise:
                q *= 1 + noise * rng.gauss(0.0, 1.0)
            rows.append((float(times[step]), q, p["Di"], p["b"], p["Dinf"], p["n"], p["T"]))
    return name, rows


class DeclineParameters(nn.Module):
    """The parameters of all five models for all wells, fitted during training.

    The models have their own sets (``qi`` and their own constants, constant
    over a well's history), so the models are compared at their own best fit
    instead of competing for a shared ``Di``. Values stay inside FIT_RANGES:
    ``low + (high - low) * sigmoid(raw)`` for b and n, ``low * (high / low) **
    sigmoid(raw)`` for the others; raw 0 is the middle of the range.
    """

    def __init__(self, num_wells: int):
        super().__init__()
        self.names = [(m, key) for m, name in enumerate(DECLINE_MODELS) for key in FIT_RANGES[name]]
        self.raw = nn.Parameter(torch.zeros(num_wells, len(self.names)))

    def forward(self) -> list[dict[str, torch.Tensor]]:
        """One dict per model, parameter name to a (num_wells,) tensor."""
        out: list[dict[str, torch.Tensor]] = [{} for _ in DECLINE_MODELS]
        for j, (m, key) in enumerate(self.names):
            low, high = FIT_RANGES[DECLINE_MODELS[m]][key]
            s = torch.sigmoid(self.raw[:, j])
            out[m][key] = low + (high - low) * s if key in LINEAR_PARAMETERS else low * (high / low) ** s
        return out


class ModelSelector(nn.Module):
    """The weights of the five residuals: a softmax over one learnt logit per model.

    The weights are one distribution per run (one model generated the data),
    start uniform, and are trained towards the model whose residual the data
    let it zero (see train).
    """

    def __init__(self, num_models: int = len(DECLINE_MODELS)):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_models))

    def forward(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)


def time_derivative(q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """dq/dt element by element: ``out[s, b] = dq[s, b] / dt[s, b]``.

    ``t`` must be the tensor ``q`` was computed from, with ``requires_grad``,
    and ``q[s, b]`` must depend on ``t`` only through ``t[s, b]``: true of an
    analytic curve, and of PINNTransformer's query stream
    (``model(t, context=times)``). One backward pass then gives all the
    derivatives at once, because the gradient of ``q.sum()`` at ``t[s, b]`` is
    ``dq[s, b] / dt[s, b]``. The graph is kept (``create_graph``) so a loss on
    the derivative trains the model.

    Raises ``ValueError`` instead of returning zeros if ``t`` cannot carry a
    derivative: a silent zero lost the physics term before.
    """
    if not t.requires_grad:
        raise ValueError(
            "time_derivative: t does not require grad, so dq/dt cannot be taken; "
            "pass the tensor the model read, created with requires_grad_(True)"
        )
    if t.numel() != q.numel():
        raise ValueError(
            f"time_derivative: q {tuple(q.shape)} and t {tuple(t.shape)} must hold one time per rate"
        )
    (grad,) = torch.autograd.grad(q.sum(), t, create_graph=True, allow_unused=True)
    if grad is None:
        raise ValueError("time_derivative: q does not depend on t")
    return grad.reshape(q.shape)


class PDEConstraints(nn.Module):
    """The five residuals: dq/dt minus the closed-form slopes of the five decline models.

    Residual k vanishes along model k's curve with model k's parameters, and
    stays above zero along the others' curves for any parameters in
    FIT_RANGES (tests/test_residuals.py).
    """

    def __init__(self, num_pde_constraints: int = len(DECLINE_MODELS)):
        super().__init__()
        if num_pde_constraints != len(DECLINE_MODELS):
            raise ValueError(f"PDEConstraints: there are {len(DECLINE_MODELS)} decline models, not {num_pde_constraints}")

    def forward(self, dq_dt: torch.Tensor, t: torch.Tensor, params: list) -> torch.Tensor:
        # dq_dt, t: (seq_len, batch_size); params: one dict per model of
        # (batch_size,) tensors, as DeclineParameters returns them.
        _require_same_shape("dq/dt", dq_dt, "t", t)
        return torch.stack(
            [dq_dt - decline_slope(k, t, params[k]) for k in range(len(DECLINE_MODELS))], dim=-1
        )


def residual_errors(residuals: torch.Tensor, mask: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Mean squares of the residuals over the masked steps, divided by ``scale``: shape (5,)."""
    weight = mask.unsqueeze(-1)
    return (residuals.pow(2) * weight).sum(dim=(0, 1)) / weight.sum().clamp_min(1) / scale


def decline_misfits(t: torch.Tensor, rates: torch.Tensor, params: list, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared errors of the models' closed-form curves against the rates, over the masked steps: shape (5,).

    The curves are the integrals of the residuals' equations, so a model
    whose residual the data can zero also matches them here, and only it.
    """
    errors = [((decline_rate(k, t, params[k]) - rates).pow(2) * mask).sum() for k in range(len(DECLINE_MODELS))]
    return torch.stack(errors) / mask.sum().clamp_min(1)


def _require_same_shape(name_a: str, a: torch.Tensor, name_b: str, b: torch.Tensor) -> None:
    """Refuse an element-wise comparison that would broadcast instead of match."""
    if a.shape != b.shape:
        raise ValueError(
            f"{name_a} {tuple(a.shape)} and {name_b} {tuple(b.shape)} must have the same shape"
        )


class PINNLoss(nn.Module):
    """The reported loss: the rate's mean squared error plus the weighted residual errors."""

    def __init__(self, lambda_data: float = 1.0, lambda_pde: float = 1.0):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_pde = lambda_pde
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_q: torch.Tensor, observed_q: torch.Tensor, weights: torch.Tensor, errors: torch.Tensor) -> torch.Tensor:
        # predicted_q, observed_q: (seq_len, batch_size) rates; weights, errors:
        # (5,), one per decline model.
        _require_same_shape("predicted q", predicted_q, "observed q", observed_q)
        _require_same_shape("model weights", weights, "residual errors", errors)
        data_loss = self.mse_loss(predicted_q, observed_q)
        pde_loss = (weights * errors).sum()
        return self.lambda_data * data_loss + self.lambda_pde * pde_loss


class PINNTransformer(nn.Module):
    """The rate network: q at query times, well by well, from the observed times."""

    def __init__(self, input_dim: int, num_layers: int, d_model: int, num_heads: int,
                 dim_feedforward: int, num_wells: int = 1, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        # 2a. Embedding layer for input features, plus a learnt code per well:
        # the wells share their times, and the code tells their curves apart.
        self.embedding = nn.Linear(input_dim, d_model)
        self.well_codes = nn.Embedding(num_wells, d_model)

        # 2b. Transformer Encoder
        self.transformer_encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len
        )

        # 2c. Linear layer for predicting the rate q
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, context: torch.Tensor = None) -> torch.Tensor:
        """Predict the rate at the times ``x``: shape (seq_len, batch_size, 1).

        x: (seq_len, batch_size, input_dim), one column per well. With
        ``context`` (the observed times, same shape), ``x`` holds query times:
        position ``s`` of ``x`` attends to the context's keys and values and
        carries no positional encoding, so the rate at ``s`` depends on ``x``
        only through ``x[s]``. The gradient with respect to ``x`` is then the
        slope of the predicted decline at the query times, with the
        observations held fixed: one backward pass (``time_derivative``)
        instead of one per time.
        """
        codes = self.well_codes(torch.arange(x.size(1), device=x.device))
        embedded_x = self.embedding(x) + codes
        embedded_context = None if context is None else self.embedding(context) + codes
        encoder_output = self.transformer_encoder(embedded_x, mask, context=embedded_context)
        return self.output_head(encoder_output)


from perd_worker import workflow, WorkflowStreamInput, WorkflowStreamOutput

#: Learning rates of the decline parameters and of the model logits; the
#: rate network uses train's learning_rate.
PARAMETER_LEARNING_RATE = 0.05
SELECTOR_LEARNING_RATE = 0.1
#: Adam steps of the decline parameters per epoch: the closed-form fits are
#: cheap next to the network, and converge in a few hundred steps.
DECLINE_FIT_STEPS = 5
#: Share of the epochs during which the weights stay uniform while all
#: models fit their parameters, before the logits start to learn.
WARMUP_SHARE = 1 / 3


@workflow.bi_di
async def train(
    inputStream: WorkflowStreamInput[float, float, float, float, float, float, float],
    batch_size: int = 4,
    epochs: int = 100,
    output_dim: int = 6,
    num_pde_constraints: int = 5,
    d_model: int = 64,
    num_heads: int = 4,
    dim_feedforward: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    lambda_data: float = 1.0,
    lambda_pde: float = 1.0,
    report_every: int = 10,
) -> WorkflowStreamOutput[int, float, str, float, float, float, float, float]:
    """Train the PINN on decline curves and identify the decline model behind them.

    Stream in one item per time step: ``(t, q, Di, b, Dinf, n, T)``, the
    time, the observed rate and the generating parameters
    (``synthetic_declines`` writes them; train does not read them). Items
    are interleaved into ``batch_size`` wells: item ``i`` is time step
    ``i // batch_size`` of well ``i % batch_size``. Times and rates are
    scaled by their largest values, so any units work.

    The network fits the rates, and the parameters of all five decline
    models (exponential, harmonic, hyperbolic, Weibull, power-law
    exponential) are fitted well by well to its slope. A softmax over five
    learnt logits weights the models' residuals: it stays uniform for the
    first third of the epochs, then learns towards the model whose residual
    is smallest relative to the others, ideally 1 for the generating model
    and 0 for the rest.

    Streams out ``(epoch, loss, model, w_exponential, w_harmonic,
    w_hyperbolic, w_weibull, w_power_law_exponential)`` on the first epoch
    and then once per ``report_every`` epochs: ``model`` is the name of the
    model with the largest weight.

    Parameters
    ----------
    batch_size : int
        Wells per run, interleaved in the stream (default 4).
    epochs : int
        Number of training epochs (default 100).
    output_dim, num_pde_constraints : int
        The stream's target columns (6) and the decline models (5); other
        values are refused.
    learning_rate : float
        Adam step size of the rate network (default 0.001).
    lambda_data, lambda_pde : float
        Weights of the data-fit and physics-residual loss terms.
    report_every : int
        Epochs between two output items (default 10).
    """

    batch_size = int(batch_size)
    epochs = int(epochs)
    output_dim = int(output_dim)
    num_pde_constraints = int(num_pde_constraints)
    d_model = int(d_model)
    num_heads = int(num_heads)
    dim_feedforward = int(dim_feedforward)
    num_layers = int(num_layers)
    dropout = float(dropout)
    learning_rate = float(learning_rate)
    lambda_data = float(lambda_data)
    lambda_pde = float(lambda_pde)
    if output_dim != 6 or num_pde_constraints != 5:
        raise ValueError(
            "train: output_dim must be 6 (q, Di, b, Dinf, n, T) and num_pde_constraints "
            f"5 (one per decline model); got {output_dim} and {num_pde_constraints}"
        )

    data = []
    async for x, q, Di, b, Dinf, n, T in inputStream:
        data.append((float(x), float(q), float(Di), float(b), float(Dinf), float(n), float(T)))

    if not data:
        raise ValueError("train needs at least one (t, q, Di, b, Dinf, n, T) item")
    data_t = torch.tensor(data, dtype=torch.float32)  # (seq_len, 7)
    # Fold rows into (seq_len // batch_size, batch_size, 7); trailing rows
    # that do not fill a batch are dropped rather than crashing the reshape.
    usable = (data_t.size(0) // batch_size) * batch_size
    if usable == 0:
        raise ValueError(
            f"train received {data_t.size(0)} item(s) but batch_size={batch_size}; "
            "stream at least batch_size items"
        )
    data_t = data_t[:usable].reshape(usable // batch_size, batch_size, 7)

    # Times in units of the longest time, rates in units of their well's
    # largest rate.
    times = data_t[:, :, :1] / data_t[:, :, 0].abs().max().clamp_min(1e-12)
    rates = data_t[:, :, 1] / data_t[:, :, 1].abs().amax(dim=0).clamp_min(1e-12)
    t = times[:, :, 0]
    # The residuals skip t <= 0, where two of the models fall infinitely fast.
    collocation = (t > 0).float()

    model = PINNTransformer(
        input_dim=1,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        num_wells=batch_size,
        dropout=dropout,
    )
    decline_parameters = DeclineParameters(num_wells=batch_size)
    selector = ModelSelector()
    pde_constraints_module = PDEConstraints(num_pde_constraints)
    pinn_loss_fn = PINNLoss(lambda_data=lambda_data, lambda_pde=lambda_pde)
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": learning_rate},
        {"params": selector.parameters(), "lr": SELECTOR_LEARNING_RATE},
    ])
    fit_optimizer = torch.optim.Adam(decline_parameters.parameters(), lr=PARAMETER_LEARNING_RATE)
    warmup = int(epochs * WARMUP_SHARE)

    print(f"Starting PINN training for {epochs} epochs...")
    for epoch in range(epochs):
        # 1. Fit all models' parameters to the rates, well by well, through
        # its closed-form curve. All five fit unweighted, so a model the
        # weights have not chosen yet still reaches its best fit.
        for _ in range(DECLINE_FIT_STEPS):
            fit_optimizer.zero_grad()
            misfits = decline_misfits(t, rates, decline_parameters(), collocation)
            misfits.sum().backward()
            fit_optimizer.step()
        params = [{key: value.detach() for key, value in p.items()} for p in decline_parameters()]
        misfits = misfits.detach()

        # 2. The network: the rates plus the residuals of the fitted models,
        # weighted by the current weights.
        model.train()
        optimizer.zero_grad()
        # The observed times again, as a leaf that can carry dq/dt: the query
        # stream reads it and the context stream reads `times` (same values).
        t_query = times.clone().requires_grad_(True)
        q = model(t_query, context=times)[:, :, 0]
        dq_dt = time_derivative(q, t_query)
        slope_scale = residual_errors(dq_dt.detach().unsqueeze(-1), collocation, torch.tensor(1.0))[0].clamp_min(1e-12)
        errors = residual_errors(pde_constraints_module(dq_dt, t, params), collocation, slope_scale)
        weights = selector()
        total_loss = pinn_loss_fn(q, rates, weights.detach(), errors)

        # 3. The weights: after the warm-up, the logits learn towards the
        # model that fits best. They read the logarithms of the misfits: the
        # gradient then depends on misfit ratios, not on the errors' scale,
        # and a model five times better than the runner-up wins as surely as
        # one a hundred times better than the rest.
        objective = total_loss
        if epoch >= warmup:
            objective = objective + (weights * torch.log(misfits.clamp_min(1e-12))).sum()

        objective.backward()
        optimizer.step()

        if (epoch + 1) % max(int(report_every), 1) == 0 or epoch == 0:
            final_weights = selector().detach()
            loss = total_loss.item()
            chosen = DECLINE_MODELS[int(final_weights.argmax())]
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}, model: {chosen}")
            yield (epoch + 1, loss, chosen, *[float(w) for w in final_weights])
