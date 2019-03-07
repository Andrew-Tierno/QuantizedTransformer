import torch
import torch.nn as nn
import linear
import torch.nn

if __name__ == "__main__":
    torch.manual_seed(1)
    x = torch.rand((3,5,2))
    lin = linear.Linear(2, 6, True)
    out = lin(x)
    avg = out.sum()
    print("x: ", x)
    print("out: ", out)
    print("avg: ", avg)
    avg.backward()