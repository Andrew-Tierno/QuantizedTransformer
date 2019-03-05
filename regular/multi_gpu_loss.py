import torch
import torch.nn as nn
from torch.autograd import Variable

class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size
        
    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, 
                                                devices=self.devices)
        criterion = nn.parallel.replicate(self.criterion,
                                          devices=self.devices)
        out_scatter = nn.parallel.scatter(out, 
                                          target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, 
                                      target_gpus=self.devices)
        normalize = normalize.to(torch.float32)
        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i+chunk_size].data, 
                                    requires_grad=self.opt is not None)] 
                           for o in out_scatter]
            if len(generator) != len(out_column):
                generator = nn.parallel.replicate(self.generator,
                                                  devices=self.devices[0:len(out_column)])
            gen = nn.parallel.parallel_apply(generator, out_column)
           
            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)), 
                  t[:, i:i+chunk_size].contiguous().view(-1)) 
                 for g, t in zip(gen, targets)]
            if len(y) != len(criterion):
                criterion = nn.parallel.replicate(self.criterion,
                                                  devices=self.devices[0:len(y)])
            loss = nn.parallel.parallel_apply(criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, 
                                   target_device=self.devices[0])
#            print("l: ", l)
#            print("l_sum: ", l.sum())
            l = l.sum() / normalize
#            print("l2: ", l)
            total += l.item()
#            print("total: ", total)
#            print("total_norm: ", total * normalize)

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())
#       print("final_total: ", total)
#        print("final_total_norm: ", total * normalize)
#        print("final_total_dtype: ", total.dtype)
#        print("final_norm_dtype: ", normalize.dtype)
#        print("final_prod_dtype: ", (total * normalize).dtype)
        # Backprop all loss through transformer.            
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, 
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize
