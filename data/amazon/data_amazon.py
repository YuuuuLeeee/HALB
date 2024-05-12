from transformers import AutoTokenizer
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import json
from collections import defaultdict

np.random.seed(7)

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    label_ids = []
    label_dict = {}
    hiera = defaultdict(set)
    with open('amazon.taxonomy', 'r') as f:
        label_dict['Root'] = -1
        for line in f.readlines():
            line = line.strip().split('\t')
            for i in line[1:]:
                if i not in label_dict:
                    label_dict[i] = len(label_dict) - 1
                hiera[label_dict[line[0]]].add(label_dict[i])
        label_dict.pop('Root')
        hiera.pop(-1)
    value_dict = {i: tokenizer.encode(v.lower(), add_special_tokens=False) for v, i in label_dict.items()}
    torch.save(value_dict, 'bert_value_dict.pt')
    torch.save(hiera, 'slot.pt')

    source = []
    labels = []
    with open('amazon_train.json', 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            source.append(tokenizer.encode(line['doc_token'].strip().lower(), truncation=True))
            tem=[]
            for i in line['doc_label']:
                if i != 'unknown':
                    tem.append(label_dict[i])
            labels.append(tem)
    train_len=len(source)
    with open('amazon_test.json', 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            source.append(tokenizer.encode(line['doc_token'].strip().lower(), truncation=True))
            tem = []
            for i in line['doc_label']:
                if i != 'unknown':
                    tem.append(label_dict[i])
            labels.append(tem)



    with open('tok.txt', 'w') as f:
        for s in source:
            f.writelines(' '.join(map(lambda x: str(x), s)) + '\n')
    with open('Y.txt', 'w') as f:
        for s in labels:
            one_hot = [0] * len(label_dict)
            for i in s:
                one_hot[i] = 1
            f.writelines(' '.join(map(lambda x: str(x), one_hot)) + '\n')

    from fairseq.binarizer import Binarizer
    from fairseq.data import indexed_dataset
    for data_path in ['tok', 'Y']:
        offsets = Binarizer.find_offsets(data_path + '.txt', 1)
        ds = indexed_dataset.make_builder(
            data_path + '.bin',
            impl='mmap',
            vocab_size=tokenizer.vocab_size,
        )
        Binarizer.binarize(
            data_path + '.txt', None, lambda t: ds.add_item(t), offset=0, end=offsets[1], already_numberized=True,
            append_eos=False
        )
        ds.finalize(data_path + '.idx')
    #split the dataset into train, valuation, and test sets
    train = [i for i in range(train_len)]
    test=[i for i in range(train_len,len(source),1)]
    train, val = train_test_split(train, test_size=0.1, random_state=0)
    torch.save({'train': train, 'val': val, 'test': test}, 'split.pt')