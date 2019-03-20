import torch.nn as nn
from layer_norm import LayerNorm
from quantizer import Quantize

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        #TODO: Maybe use BatchNorm?
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return Quantize.apply(Quantize.apply(x) + self.dropout(sublayer(self.norm(x))))
