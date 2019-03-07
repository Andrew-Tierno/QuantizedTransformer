import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from quantizer import LinearQuant
import torch.nn.init as init
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, bitwidth=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.zeros((out_features, in_features)))
        self.bitwidth = Variable(torch.tensor([bitwidth]), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def apply_linear(self, x, weight, bitwidth, bias=None):
        weight = LinearQuant.apply(weight, bitwidth)
        if bias is not None:
            bias = LinearQuant.apply(torch.jit._unwrap_optional(bias), bitwidth)
        if x.dim() == 2 and bias is not None:
        # fused op is marginally faster
            ret = torch.addmm(bias, x, weight.t())
        else:
            output = x.matmul(weight.t())
            if bias is not None:
                output += bias
            ret = output
        return LinearQuant.apply(ret, bitwidth)
    def forward(self, x):
        return self.apply_linear(x, self.weight, self.bitwidth, self.bias)

