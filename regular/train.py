import time
import torch
import torch.nn as nn

from tqdm import tqdm

from dataloader import generate_dataloaders
from model import make_model
from label_smoothing import LabelSmoothing
from batch_iterator import BatchIterator, rebatch
from noamopt import NoamOpt
from multi_gpu_loss import MultiGPULossCompute
from greedy_decoding import greedy_decode

import os


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    orig_start = time.time()
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            elapsed_orig = time.time() - orig_start
            print("Epoch Step: %d Loss: %f TPS: %f Batch Time Elapsed: %d, Total Time Elapsed: %d" %
                  (i, loss / batch.ntokens, tokens / elapsed, elapsed, elapsed_orig))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

def get_unique_folder(root, prefix):
    idx = 1
    for folder in os.listdir(root):
        if folder.startswith(prefix):
            idx += 1
    return os.path.join(root, prefix + str(idx))


def train():
    print("Loading data...")
    SRC, TGT, train, val, test = generate_dataloaders()

    devices = [0, 1, 2, 3]
    pad_idx = TGT.vocab.stoi["<blank>"]
    print("Making model...")
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model.cuda()
    criterion = LabelSmoothing(
        size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 12000
    train_iter = BatchIterator(train, batch_size=BATCH_SIZE, device=torch.device(0),
                               repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                               batch_size_fn=batch_size_fn, train=True)
    valid_iter = BatchIterator(val, batch_size=BATCH_SIZE, device=torch.device(0),
                               repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                               batch_size_fn=batch_size_fn, train=False)
    model_par = nn.DataParallel(model, device_ids=devices)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    folder = get_unique_folder("./models/", "model")
    if not(os.path.exists(folder)):
        os.mkdir(folder)
    for epoch in tqdm(range(10)):
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter),
                  model_par,
                  MultiGPULossCompute(model.generator, criterion,
                                      devices=devices, opt=model_opt))
        model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                         model_par,
                         MultiGPULossCompute(model.generator, criterion,
                                             devices=devices, opt=None))
        torch.save(model.state_dict(), os.path.join(folder, "model.bin." + str(epoch)))
        print(loss)

    for i, batch in enumerate(valid_iter):
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask,
                            max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
        print("Translation:", end="\t")
        for i in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == "</s>":
                break
            print(sym, end=" ")
        print()
        print("Target:", end="\t")
        for i in range(1, batch.trg.size(0)):
            sym = TGT.vocab.itos[batch.trg.data[i, 0]]
            if sym == "</s>":
                break
            print(sym, end=" ")
        print()
        break

if __name__ == "__main__":
    train()
