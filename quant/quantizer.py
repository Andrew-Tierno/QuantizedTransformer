import torch
import torch.nn as nn

class LinearQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bitwidth):
        min_val = x.min()
        max_val = x.max()
        ctx.save_for_backward(x, bitwidth, min_val, max_val)
        x = (x - min_val) / (max_val - min_val)
        factor = torch.Tensor([1]) << (bitwidth-1)
        return torch.round(x * factor)

    @staticmethod
    def backward(ctx, grad_out):
        x, bitwidth, min_val, max_val = ctx.saved_variables
        grad_x = grad_out * (x.norm(dim=-1) < 2**(bitwidth-1)).unsqueeze(-1).to(x.dtype)
        return grad_x, None
