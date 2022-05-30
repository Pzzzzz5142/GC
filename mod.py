from inspect import Parameter
from typing import Any
import torch
import torch.nn as nn
from torch.nn import init
import math
from torch.utils.cpp_extension import load

cov1d_cpp = load(name="cov1d_cpp", sources=["source/gc.cpp"])


class GCFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel):
        output = cov1d_cpp.forward(input, kernel)
        ctx.save_for_backward(input, kernel)
        return output


class GC(torch.nn.modules):
    def __init__(self, kernal_size, weight=None, device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if weight != None:
            assert weight.shape[-1] == kernal_size
            self.weight = weight
        else:
            self.weight = Parameter(torch.empty((1, 1, kernal_size), **factory_kwargs))
            self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        return GCFunction.apply(input, self.weight)
