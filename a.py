import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
from mod import GC

torch.random.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cov = nn.Conv1d(1, 1, 3, padding=0, device=device)
cov_cus = GC(3, cov.weight, device=device)

t_in = torch.randn(5, 1, 10, device=device)
t_out = cov(t_in)
t_out2 = cov_cus(t_in)

print(t_out, t_out2)

