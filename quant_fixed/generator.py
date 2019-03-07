import torch.nn as nn
import linear

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = linear.Linear(d_model, vocab)

    def forward(self, x):
        return nn.functional.log_softmax(self.proj(x), dim=-1)