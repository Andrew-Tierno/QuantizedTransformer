import argparse
import torch
from quant.greedy_decoding import greedy_decode
from quant.dataloader import generate_dataloaders
from tqdm import tqdm
from batch_iterator import BatchIterator


def log(text, log_file):
    print(text)
    log_file.write(text + "\n")

if __name__ == "__main__":
    BATCH_SIZE = 12000
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('log_name')

    args = parser.parse_args()
    model_file = open(args.model_name)
    log_file = open(args.log_name, 'w')

    print("Loading data...")
    SRC, TGT, train, val, test = generate_dataloaders("./data_processed/")
    test_iter = BatchIterator(val, batch_size=BATCH_SIZE, device=torch.device(0),
                               repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                               batch_size_fn=batch_size_fn, train=False)
    print("Loading model...")
    model = torch.load(model_file)
    print("Generating test output...")
    log("Testing model stored at " + args.model_name + ".", log_file)
    for i, batch in tqdm(enumerate(test_iter)):
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask,
                            max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
        for i in range(0, out.size(0)):
            for j in range(1, out.size(1)):
                sym = TGT.vocab.itos[out[i, j]]
                if sym == "</s>":
                    break
                log_file.write(sym)
            log_file.write("\n")
    
        
        
