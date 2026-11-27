import sys
from pathlib import Path

import torch

# The workflow and the page import as top-level modules, as on PERD.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Two threads, as on the small compute class (2 vCPU): the model is small
# enough that more threads cost more in coordination than they save.
torch.set_num_threads(2)
