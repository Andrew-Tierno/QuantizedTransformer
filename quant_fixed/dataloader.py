from torchtext import data, datasets
from pyvi import ViTokenizer
import spacy

spacy_en = spacy.load('en')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_vi(text):
    return ViTokenizer.tokenize(text).split()

curr_print = 0
def myfilter(x):
    MAX_LEN = 100
    return 'src' in vars(x) and 'trg' in vars(x) and len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

def generate_dataloaders():
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_vi, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, 
        eos_token = EOS_WORD, pad_token=BLANK_WORD)
    MAX_LEN = 100
    train, val, test = data.TabularDataset.splits(
        path="../data_processed/", train='train.tsv', test='test2013.tsv',
        validation='dev.tsv', fields=[('src',SRC), ('trg',TGT)], 
        format='tsv', filter_pred=myfilter)#lambda x: len(vars(x)['src']) <= MAX_LEN and 
#            len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    return (SRC, TGT, train, val, test)
