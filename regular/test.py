import argparse
import torch
from greedy_decoding import greedy_decode
from dataloader import generate_dataloaders
from tqdm import tqdm
from batch_iterator import BatchIterator
from model import make_model

def log(text, log_file):
    print(text)
    log_file.write(text + "\n")

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

if __name__ == "__main__":
    BATCH_SIZE = 12000
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('log_name')

    args = parser.parse_args()
    model_file = open(args.model_name, 'rb')
    log_file = open(args.log_name, 'w+')

    print("Loading data...")
    SRC, TGT, train, val, test = generate_dataloaders("../data_processed/")
    test_iter = BatchIterator(test, batch_size=BATCH_SIZE, device=torch.device(0),
                               repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                               batch_size_fn=batch_size_fn, train=False)
    print("Loading model...")
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model.load_state_dict(torch.load(args.model_name))
    model.cuda()
    model.eval()
    print("Generating test output...")
    log("Testing model stored at " + args.model_name + ".", log_file)
    for k, batch in tqdm(enumerate(test_iter)):
        print("Batch Size Orig: ", batch.src.size())
#        print("k: ", k)
        src_orig = batch.src.transpose(0, 1)
        trg_orig = batch.trg.transpose(0, 1)
        print("Batch Size Final: ", src_orig.size())
#        print("src: ", src)
#        print("src_size: ", src.size())
        for m in tqdm(range(0, len(src_orig), 1)):
            src = src_orig[m:(m+1)]
            trg = trg_orig[m:(m+1)]
            src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
            out = greedy_decode(model, src, src_mask,
                            max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
            for i in range(0, out.size(0)):
                for j in range(1, out.size(1)):
                    sym = TGT.vocab.itos[out[i, j]]
                    if sym == "</s>":
                        break
                    log_file.write(sym)
                    log_file.write(" ")
            log_file.write("\t||\t")
            for i in range(trg.size(0)):
                for j in range(1, trg.size(1)):
                    sym = TGT.vocab.itos[trg[i, j]]
                    if sym == "<unk>":
                        sym = "<ood>"
                    if sym == "</s>":
                        break
                    log_file.write(sym)
                    log_file.write(" ")
            log_file.write("\n")
