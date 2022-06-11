from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from mod import GC

torch.random.manual_seed(42)

kernel_size=3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cov = nn.Conv1d(1, 1, kernel_size, padding=1, device=device, bias=False)
cov_cus = GC(kernel_size, cov.weight, device=device,)

t_in = torch.randn(256, 1, 512, device=device)
t_out = cov(t_in)
t_out2 = cov_cus(t_in)

assert t_out.allclose(t_out2,atol=1e-07)

t0 = benchmark.Timer(stmt="cov(t_in)", globals={"t_in": t_in, "cov": cov})

t1 = benchmark.Timer(stmt="cov_cus(t_in)", globals={"t_in": t_in, "cov_cus": cov_cus})

print(t0.timeit(100))
print(t1.timeit(100))
