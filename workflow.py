"""
PINN Workflow
"""

import torch
import torch.nn as nn
import math

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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention block with residual connection and layer normalization
        attn_output = self.self_attn(x, x, x, mask)
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
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

class PDEConstraints(nn.Module):
    def __init__(self, num_pde_constraints: int):
        super().__init__()
        self.num_pde_constraints = num_pde_constraints
        # print(f"PDEConstraints initialized with {num_pde_constraints} constraints.")
        if num_pde_constraints != 5:
            print(f"Warning: Expected 5 PDE constraints for the given equations, but got {num_pde_constraints}.")

    def forward(self, model_output: torch.Tensor, input_data: torch.Tensor) -> torch.Tensor:
        # model_output: (seq_len, batch_size, output_dim=6) where output_dim is q, Di, b, Dinf, n, T
        # input_data: (seq_len, batch_size, input_dim=1) which is 't'

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

        # Calculate dq_dt using torch.autograd.grad for each element (element-wise derivative)
        # Note: This loop is extremely slow for large batches/seq_len. Vectorized grad is harder but possible.
        # Keeping loop as per original notebook for now.
        dq_dt_values = torch.zeros_like(q)
        for s_idx in range(seq_len):
            for b_idx in range(batch_size):
                q_scalar = q[s_idx, b_idx]
                t_scalar = t[s_idx, b_idx]

                # Compute scalar derivative d(q_scalar)/d(t_scalar)
                if t_scalar.requires_grad:
                    grad_output = torch.autograd.grad(
                        outputs=q_scalar,
                        inputs=t_scalar,
                        create_graph=True, # Needed to allow backprop through this derivative computation
                        allow_unused=True  # If q_scalar doesn't depend on t_scalar, it will return None
                    )[0]
                else:
                    grad_output = None

                if grad_output is None:
                    dq_dt_values[s_idx, b_idx] = torch.tensor(0.0, device=q.device, dtype=q.dtype)
                else:
                    dq_dt_values[s_idx, b_idx] = grad_output
        dq_dt = dq_dt_values

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

class PINNLoss(nn.Module):
    def __init__(self, lambda_data: float = 1.0, lambda_pde: float = 1.0):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_pde = lambda_pde
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_y: torch.Tensor, true_y: torch.Tensor, predicted_weights: torch.Tensor, pde_residuals: torch.Tensor) -> torch.Tensor:
        # 1. Calculate data loss (e.g., MSE)
        data_loss = self.mse_loss(predicted_y[:, :, 0].unsqueeze(-1), true_y)

        # 2. Calculate PDE loss
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (seq_len, batch_size, input_dim)

        # 3b. Pass input x through the initial embedding layer
        embedded_x = self.embedding(x)

        # 3c. Pass the embedded input through the TransformerEncoder
        encoder_output = self.transformer_encoder(embedded_x, mask)

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

def train(
    X,
    Y,
    epochs: int = 100,
    output_dim = 6,
    num_pde_constraints = 5,
    d_model = 64,
    num_heads = 4,
    dim_feedforward = 128,
    num_layers = 2,
    dropout = 0.1,
    learning_rate = 0.001,
    lambda_data = 1.0,
    lambda_pde = 1.0,
    live_mode=False
) -> dict:
    """
    ## pinn
    
    Runs a Physics-Informed Neural Network (PINN) Transformer training workflow.
    
    ### **parameters**
    
    epochs : int
        Number of training epochs (default: 100)
        
    ### **returns**
    
    dict
        Training results including final loss.
    """

    pde_input_dummy = X
    input_dim = X.shape[-1]

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

        predicted_y, adaptive_weights = model(X)
        pde_residuals = pde_constraints_module(predicted_y, pde_input_dummy)

        total_loss = pinn_loss_fn(
            predicted_y=predicted_y,
            true_y=Y,
            predicted_weights=adaptive_weights,
            pde_residuals=pde_residuals
        )

        total_loss.backward(retain_graph=True)
        optimizer.step()

        final_loss = total_loss.item()
        loss_history.append(final_loss)
        
        if live_mode and ((epoch + 1) % 10 == 0 or epoch == 0):
            # pending functionality to return live plotting data over websocket connection
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {final_loss:.4f}")

    print("Training loop finished.")
    return {
        "status": "success", 
        "final_loss": final_loss, 
        "epochs": epochs,
        "loss_history": loss_history
    }
