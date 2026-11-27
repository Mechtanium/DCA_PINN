"""DCA_PINN: a Physics-Informed Transformer for decline-curve analysis, as a
PERD workflow.

The module-level ``workflow`` registry (from ``perd_worker``) is what PERD
serves: publish this repository on the website (or with ``python -m perd
workflow publish --git <this repo>``), start a workstation on it, and call
``ws.workflows.dca_pinn.train(...)`` from the ``perd`` library.
"""

import math

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
        x = self.pos_encoder(x)
        if context is not None:
            # Two streams: the context runs through the layers as ordinary
            # self-attention, and the query stream x reads the context's state
            # at the entry of the same layer.
            context = self.pos_encoder(context)
            for i, layer in enumerate(self.layers):
                query = layer(x, mask, context=context)
                if i + 1 < len(self.layers):
                    context = layer(context, mask)
                x = query
            return self.norm(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x) # Final layer norm, often applied after the last encoder layer
        return x

class SelfAdaptiveWeightPredictionHead(nn.Module):
    def __init__(self, d_model: int, num_pde_constraints: int):
        super().__init__()
        self.linear = nn.Linear(d_model, num_pde_constraints)
        # Using Softplus to ensure non-negative weights
        self.activation = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch_size, d_model)
        # Apply linear layer
        weights = self.linear(x)
        # Apply activation to ensure non-negative weights
        weights = self.activation(weights)
        # Output shape: (seq_len, batch_size, num_pde_constraints)
        return weights

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
    def __init__(self, num_pde_constraints: int):
        super().__init__()
        self.num_pde_constraints = num_pde_constraints
        # print(f"PDEConstraints initialized with {num_pde_constraints} constraints.")
        if num_pde_constraints != 5:
            print(f"Warning: Expected 5 PDE constraints for the given equations, but got {num_pde_constraints}.")

    def forward(self, model_output: torch.Tensor, input_data: torch.Tensor) -> torch.Tensor:
        # model_output: (seq_len, batch_size, output_dim=6) where output_dim is q, Di, b, Dinf, n, T
        # input_data: (seq_len, batch_size, input_dim=1) which is 't': the
        # tensor the model's query stream consumed, with requires_grad.

        seq_len, batch_size, output_dims = model_output.shape
        if input_data.shape[-1] != 1 or output_dims != 6:
            raise ValueError(f"Expected input_data (t) to have last dim 1 and model_output to have last dim 6. Got input {input_data.shape} and output {model_output.shape}")

        # Extract t, q, Di, b, Dinf, n, T
        t = input_data[:, :, 0]  # (seq_len, batch_size)
        q = model_output[:, :, 0] # (seq_len, batch_size)
        Di = model_output[:, :, 1] # (seq_len, batch_size)
        b = model_output[:, :, 2] # (seq_len, batch_size)
        Dinf = model_output[:, :, 3] # (seq_len, batch_size)
        n = model_output[:, :, 4] # (seq_len, batch_size)
        T = model_output[:, :, 5] # (seq_len, batch_size)

        # dq[s, b]/dt[s, b], differentiated with respect to the tensor the
        # model read (a slice such as t above is a new node q never saw).
        dq_dt = time_derivative(q, input_data)

        # Extract qi (initial flow rate) - q at t=0. This will have shape (batch_size,)
        # It's extracted from the first element in the sequence for each batch.
        qi = q[0, :]

        # Expand qi to match (seq_len, batch_size) for element-wise operations with t and Di
        qi_expanded = qi.unsqueeze(0).expand(seq_len, batch_size) # (seq_len, batch_size)

        # pde_1: dq_dt + Di * qi * exp( -Di * t) = 0
        pde1_residual = dq_dt + Di * qi_expanded * torch.exp(-Di * t)

        # pde_2: dq_dt + [(Di * qi) / (1 + Di * t)] = 0
        pde2_residual = dq_dt + (Di * qi_expanded) / (1 + Di * t)

        # pde_3: dq_dt + [(Di * qi) / (1 + b * Di * t)^(1/b - 1)] = 0
        pde3_base = (1 + b * Di * t)
        pde3_exponent = (1 / b - 1)
        pde3_residual = dq_dt + (Di * qi_expanded) / torch.pow(torch.max(pde3_base, torch.tensor(1e-6)), pde3_exponent)

        # pde_4: dq_dt + (n * qi / T) * (t/T)^(n-1) * exp(-(t/T)^n)
        T_stable = torch.max(T, torch.tensor(1e-6))
        t_div_T = t / T_stable
        pde4_residual = dq_dt + (n * qi_expanded / T_stable) * torch.pow(t_div_T, n - 1) * torch.exp(-torch.pow(t_div_T, n))

        # pde_5: dq_dt - qi * ((Di * t^(-n)) - Dinf) * exp((D/(1-n)) * (t^(1-n) - 1) - Dinf * t)
        t_stable = torch.max(t, torch.tensor(1e-6))
        pde5_term1 = (Di * torch.pow(t_stable, -n)) - Dinf
        pde5_term2_exponent = (Di / torch.max(torch.tensor(1e-6), (1 - n))) * (torch.pow(t_stable, (1 - n)) - 1) - Dinf * t
        pde5_residual = dq_dt - qi * pde5_term1 * torch.exp(pde5_term2_exponent)

        # Stack residuals to get (seq_len, batch_size, num_pde_constraints)
        pde_residuals = torch.stack([pde1_residual, pde2_residual, pde3_residual, pde4_residual, pde5_residual], dim=-1)

        return pde_residuals

def _require_same_shape(name_a: str, a: torch.Tensor, name_b: str, b: torch.Tensor) -> None:
    """Refuse an element-wise comparison that would broadcast instead of match."""
    if a.shape != b.shape:
        raise ValueError(
            f"{name_a} {tuple(a.shape)} and {name_b} {tuple(b.shape)} must have the same shape"
        )


class PINNLoss(nn.Module):
    def __init__(self, lambda_data: float = 1.0, lambda_pde: float = 1.0):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_pde = lambda_pde
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_y: torch.Tensor, true_y: torch.Tensor, predicted_weights: torch.Tensor, pde_residuals: torch.Tensor) -> torch.Tensor:
        # predicted_y, true_y: (seq_len, batch_size, 6) with (q, Di, b, Dinf, n, T)
        # in the last dimension. predicted_weights, pde_residuals:
        # (seq_len, batch_size, num_pde_constraints).

        # 1. Data loss: the predicted rate against the observed rate, column 0
        # of both. The decline parameters are learnt through the residuals.
        q_pred = predicted_y[..., 0:1]
        q_true = true_y[..., 0:1]
        _require_same_shape("predicted q", q_pred, "observed q", q_true)
        data_loss = self.mse_loss(q_pred, q_true)

        # 2. Calculate PDE loss
        _require_same_shape("adaptive weights", predicted_weights, "PDE residuals", pde_residuals)
        # Square the PDE residuals
        squared_pde_residuals = pde_residuals.pow(2)

        # Element-wise multiplication with predicted_weights
        weighted_pde_loss_terms = predicted_weights * squared_pde_residuals

        # Sum across the num_pde_constraints dimension first, then average over seq_len and batch_size
        pde_loss = torch.mean(torch.sum(weighted_pde_loss_terms, dim=-1))

        # 3. Combine data loss and adaptively weighted PDE loss
        total_loss = self.lambda_data * data_loss + self.lambda_pde * pde_loss

        return total_loss

class PINNTransformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int, d_model: int,
                 num_heads: int, dim_feedforward: int, num_pde_constraints: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        # 2a. Embedding layer for input features
        self.embedding = nn.Linear(input_dim, d_model)

        # 2b. Transformer Encoder
        self.transformer_encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len
        )

        # 2c. Linear layer for predicting sequential output y
        self.output_head = nn.Linear(d_model, output_dim)

        # 2d. Self-Adaptive Weight Prediction Head
        self.weight_head = SelfAdaptiveWeightPredictionHead(d_model, num_pde_constraints)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, context: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the six outputs and the adaptive weights at the times ``x``.

        x: (seq_len, batch_size, input_dim). Without ``context`` the positions
        of ``x`` attend to each other, so the prediction at one time depends
        on all the times of its sequence.

        With ``context`` (the observed times, same shape), ``x`` holds query
        times: position ``s`` of ``x`` attends to the context's keys and
        values, so the outputs at ``s`` depend on ``x`` only through
        ``x[s]``. Passing the observed times as both gives the same values as
        ``model(x)``, and the gradient with respect to ``x`` is then the slope
        of the predicted decline at the query times with the observations held
        fixed: one backward pass (``time_derivative``) instead of one per time.
        """

        # 3b. Pass input x through the initial embedding layer
        embedded_x = self.embedding(x)
        embedded_context = None if context is None else self.embedding(context)

        # 3c. Pass the embedded input through the TransformerEncoder
        encoder_output = self.transformer_encoder(embedded_x, mask, context=embedded_context)

        # 3d. Use the encoder_output to predict the sequential output y
        predicted_y_raw = self.output_head(encoder_output)

        # Apply clamping or activation functions to predicted_y_raw components
        q_pred = predicted_y_raw[:, :, 0:1]
        Di_pred = torch.clamp(predicted_y_raw[:, :, 1:2], min=1e-6)
        b_pred = torch.clamp(predicted_y_raw[:, :, 2:3], min=1e-6, max=1.0 - 1e-6)
        Dinf_pred = torch.clamp(predicted_y_raw[:, :, 3:4], min=1e-6)
        n_pred = torch.clamp(predicted_y_raw[:, :, 4:5], min=1e-6, max=1.0 - 1e-6)
        T_pred = torch.clamp(predicted_y_raw[:, :, 5:6], min=1e-6)

        predicted_y = torch.cat([q_pred, Di_pred, b_pred, Dinf_pred, n_pred, T_pred], dim=-1)

        # 3e. Use the encoder_output to predict the adaptive_weights
        adaptive_weights = self.weight_head(encoder_output)

        # 3f. Return both predicted_y and adaptive_weights
        return predicted_y, adaptive_weights


from perd_worker import workflow, WorkflowStreamInput, WorkflowStreamOutput


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
) -> WorkflowStreamOutput[int, float]:
    """Train the PINN Transformer on a stream of decline-curve samples.

    Stream in one item per time step: ``(t, q, Di, b, Dinf, n, T)``, the
    time, the observed rate and the five DCA parameters. The data loss
    compares the predicted rate with ``q``; the network infers the five
    parameters through the physics residuals, so their columns are read but
    not compared with anything. Items are consumed in order and interleaved
    into ``batch_size`` sequences: item ``i`` is time step ``i // batch_size``
    of sequence ``i % batch_size``.

    Streams out ``(epoch, loss)`` on the first epoch and every
    ``report_every`` epochs after that, so a client can plot the loss live.

    Parameters
    ----------
    batch_size : int
        Sequences per optimisation step (default 4).
    epochs : int
        Number of training epochs (default 100).
    learning_rate : float
        Adam step size (default 0.001).
    lambda_data, lambda_pde : float
        Weights of the data-fit and physics-residual loss terms.
    report_every : int
        Emit ``(epoch, loss)`` every this many epochs (default 10).
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

    input_dim = data_t.shape[-1] - output_dim  # Assuming last 6 columns are true_y (q, Di, b, Dinf, n, T)
    X = data_t[:, :, :input_dim]
    Y = data_t[:, :, input_dim:]

    # Model instantiation
    model = PINNTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        num_pde_constraints=num_pde_constraints,
        dropout=dropout
    )

    pde_constraints_module = PDEConstraints(num_pde_constraints=num_pde_constraints)
    pinn_loss_fn = PINNLoss(lambda_data=lambda_data, lambda_pde=lambda_pde)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting PINN training for {epochs} epochs...")
    final_loss = 0.0
    loss_history = [] 

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # The observed times again, as a leaf that can carry dq/dt: the query
        # stream reads it and the context stream reads X (same values).
        t_query = X.clone().requires_grad_(True)
        predicted_y, adaptive_weights = model(t_query, context=X)
        pde_residuals = pde_constraints_module(predicted_y, t_query)

        total_loss = pinn_loss_fn(
            predicted_y=predicted_y,
            true_y=Y,
            predicted_weights=adaptive_weights,
            pde_residuals=pde_residuals
        )

        total_loss.backward()
        optimizer.step()

        final_loss = total_loss.item()
        loss_history.append(final_loss)
        
        if (epoch + 1) % max(int(report_every), 1) == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {final_loss:.4f}")
            yield epoch + 1, final_loss
