import re
import copy

def parse_train_dict(train_dict, fname):
    with open(fname) as src:
        curr_entry = {}
        for line in src:
            match = re.match("<(\w+)>", line)
            if match is not None:
                tag = match.group(1)
                line = line.replace("<" + tag + ">", "").replace("</" + tag + ">", "")
                curr_entry[tag] = line.strip()
                #print(tag)
                if tag == "url" and len(curr_entry) > 1:
                    #print("Adding id [{0}]".format(curr_entry["talkid"]))
                    train_dict[curr_entry["talkid"]] = curr_entry
                    #print(len(train_dict))
                    curr_entry = {}
            else:
                if "text" not in curr_entry:
                    curr_entry["text"] = []
                curr_entry["text"].append(line.strip())
        train_dict[curr_entry["talkid"]] = curr_entry
def preprocess_train(source_file, target_file, out_file):
    with open(source_file) as src, open(target_file) as tgt, open(out_file, 'w') as out:
        for src_line, tgt_line in zip(src, tgt):
            if (not(src_line.startswith("<"))):
                out.write("{0}\t{1}\n".format(src_line.strip(), tgt_line.strip()))
    # DESC_OPEN = "<description>"
    # DESC_CLOSE = "</description>"
    # src_dict = {}
    # tgt_dict = {}

    # parse_train_dict(src_dict, source_file)
    # print("Done")
    # print(len(src_dict))
    # parse_train_dict(tgt_dict, target_file)
    # print("Done")
    # print(len(tgt_dict))
    # print(len(src_dict))
    # idx = 1
    # with open(out_file, 'w') as out:
    #     print(src_dict.keys())
    #     for talkid in src_dict:
    #         print("Processing " + str(idx) + " (" + talkid + ")")
    #         idx += 1
    #         src_entry = src_dict[talkid]
    #         tgt_entry = tgt_dict[talkid]
    #         print("[{0}]: {1} ||||| {2}".format(talkid, src_entry['text'][0], tgt_entry['text'][0]))
    #         for src_line, tgt_line in zip(src_entry["text"], tgt_entry["text"]):
    #             out.write("{0}\t{1}\n".format(src_line, tgt_line))

def preprocess_devtest(source_file, target_file, out_file):
    with open(source_file) as src, open(target_file) as tgt, open(out_file, 'w') as out:
        for src_line, tgt_line in zip(src, tgt):
            match = re.match('<seg id=\"\d+\">', src_line)
            if match is not None:
                src_line = src_line.replace(match.group(0), '').replace("</seg>", '')
                tgt_line = tgt_line.replace(match.group(0), '').replace("</seg>", '')
                out.write("{0}\t{1}\n".format(src_line.strip(), tgt_line.strip()))

if __name__ == "__main__":
    preprocess_train("data/train.tags.en-vi.vi", "data/train.tags.en-vi.en", "data_processed/train.tsv")
    preprocess_devtest("data/IWSLT15.TED.dev2010.en-vi.vi.xml", "data/IWSLT15.TED.dev2010.en-vi.en.xml", "data_processed/dev.tsv")
    tests_en = ["data/IWSLT15.TED.tst2010.en-vi.en.xml", "data/IWSLT15.TED.tst2011.en-vi.en.xml", "data/IWSLT15.TED.tst2012.en-vi.en.xml", "data/IWSLT15.TED.tst2013.en-vi.en.xml"]
    tests_vi = ["data/IWSLT15.TED.tst2010.en-vi.vi.xml", "data/IWSLT15.TED.tst2011.en-vi.vi.xml", "data/IWSLT15.TED.tst2012.en-vi.vi.xml", "data/IWSLT15.TED.tst2013.en-vi.vi.xml"]
    for i in range(len(tests_en)):
        preprocess_devtest(tests_vi[i], tests_en[i], "data_processed/test201" + str(i) + ".tsv")