import torch
import torch.nn as nn

class LinearQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bitwidth):
        max_val = x.std()
        min_val = -x.std()
        bitwidth = bitwidth.to(device=x.device)
        ctx.save_for_backward(x, bitwidth, min_val, max_val)
        x = (x - min_val) / (max_val - min_val)
        factor = torch.Tensor([1]) << (bitwidth-1)
        return torch.round(x * factor.to(torch.float32))

    @staticmethod
    def backward(ctx, grad_out):
        x, bitwidth, min_val, max_val = ctx.saved_variables
#        print("b_d: ", bitwidth.dtype)
#        print("f_d: ", (x.norm(dim=-1) < 2**(bitwidth-1)).unsqueeze(-1).to(x.dtype).dtype)
#        print("x_d: ", x.dtype)
        factor = torch.tensor([1], device=x.device) << (bitwidth-1)
        factor = factor.to(torch.float32)
        grad_x = grad_out * (((x.norm(dim=-1) < factor)).unsqueeze(-1).to(x.dtype))
        grad_x = grad_x * (max_val - min_val) / (2 * factor)
        return grad_x, None
