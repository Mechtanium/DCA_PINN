# Workflow Template - Create Your Own Custom Workflows

Decline Curve Analysis (DCA) is one of the primary ways to estimate expected pressure declines and total recovery in pressurized porous fluid diffusion systems.

In this workflow a Physics Informed Neural Network with a Transformer-based core architecture is used to learn from DCA models in an automatic self-attention driven way how to predict DCA decline. The network learns how to predict DCA accurately by automatically deciding which DCA models to apply at which stage and in which systems.

## Prerequisites
- Python 3.7+
- pip package manager

**Install base dependencies:**
```bash
pip install --quiet torch
pip install --quiet per_datasets
```

## Using the workflow
```python
import torch
import per_datasets

seq_len = 1000
batch_size = 4
input_dim = 1
D = 2e-2

# ---------------
# Data generation
# ---------------

# t_dummy: Represents input sequence (e.g., time steps)
# For PINNs, inputs for derivative calculations must have requires_grad=True.
# Create monotonic increasing time sequences per batch from 0 to 1 with random steps.
# First value is always zero; values increase in random steps and the final value is 1.
zeros = torch.zeros(1, batch_size, input_dim) + 1e-4
increments = torch.rand(seq_len - 1, batch_size, input_dim)
cumsum = torch.cat([zeros, increments.cumsum(dim=0)], dim=0)
# Normalize each sequence so its last element equals 1 (avoid division by zero)
last = cumsum[-1:].clone()
last[last == 0] = 1.0
t_dummy = (cumsum / last)
# Enable gradient tracking for PINN derivative calculations
t_dummy.requires_grad_()

qi_args = (torch.rand(batch_size, 1) * 1000) + 2500  # Random qi between 2500 and 3500

# Compute true q using qi_args per batch and the analytic decay q = qi * exp(-D * t)
# qi_args has shape (batch_size, 1); expand to (1, batch_size, 1) to broadcast over seq_len
q_true_dummy = (qi_args.unsqueeze(0) * torch.exp(-D * t_dummy.detach())) + (torch.rand(seq_len, batch_size, input_dim) * 5)  # Adding small noise

# --------------
# DCA_PINN usage
# --------------

with per_datasets.initialize(workflows=['DCA_PINN_Workflow_ID/from/perd-website'], DISK=250, RAM=24, vCPU=8) as pds: # DISK && RAM are in GB
    results = pds.workflows.DCA_PINN.train(t_dummy, q_true_dummy, epochs=50, learning_rate = 1, live_mode=True)
    print(f"Final Loss: {results['final_loss']}")

    # Or use pds.visual if imported as pds
    pds.visual.line_plot(results, y='loss_history', title="PINN Training Loss")

## Server

This repository also provides a small Flask server exposing a `/train` route that accepts a JSON payload and runs the `train` workflow.

Build and run with Docker (example):

```bash
docker build -t dca_pinn:latest .
docker run -p 5000:5000 dca_pinn:latest
```

Example `curl` payload (replace arrays with appropriate shapes):

```bash
curl -X POST http://localhost:5000/train \
    -H "Content-Type: application/json" \
    -d '{"X": [[[0.0]]], "Y": [[[1.0]]], "epochs": 10}'
```

├── workflows/
│   ├── __init__.py                 # Main exports
│   ├── add/                        # Example workflow
│   │   ├── __init__.py
│   │   ├── workflow.py
│   │   └── requirements.txt
│   └── my_custom_workflow/         # Your custom workflow
│       ├── __init__.py
│       ├── workflow.py
│       └── requirements.txt             
└── README.md                      
```

## Running the Flask server

Local development (dev server):

```bash
python main.py
```

Production (gunicorn):

```bash
gunicorn --bind 0.0.0.0:5000 main:app
```

Docker build and run:

```bash
docker build -t workflow-server .
docker run -p 5000:5000 workflow-server
```

Example request:

```bash
curl -X POST -H "Content-Type: application/json" \
    -d '{"a":2.5,"b":3.7}' http://localhost:5000/add
```