import json
import numpy as np
import pandas as pd
from tqdm import tqdm
"""
Amazon Reference: https://www.kaggle.com/datasets/kashnitsky/hierarchical-text-classification
"""

FILE_DIR = 'Amazon/'
total_len = []
np.random.seed(7)


def get_data_from_meta(file_name):
    file = pd.read_csv(FILE_DIR + file_name)
    data = []
    for i, line in tqdm(file.iterrows()):
        sample_text = line['Text']
        if sample_text == '':
            print("something wrong with text......")
            continue
        else:
            sample_label = []
            sample_label.append(line['Cat1'])
            sample_label.append(line['Cat2'])
            sample_label.append(line['Cat3'])
            data.append({'doc_token': sample_text, 'doc_label': sample_label})
    if file_name == 'train_40k.csv':
        f = open('amazon_train.json', 'w')
    elif file_name == 'val_10k.csv':
        f = open('amazon_test.json', 'w')
    for line in data:
        line = json.dumps(line)
        f.write(line + '\n')
    f.close()


def get_hierarchy():
    f_train = open('amazon_train.json', 'r')
    train_data = f_train.readlines()
    f_train.close()
    f_test=open('amazon_test.json','r')
    test_data=f_test.readlines()
    f_test.close()

    label_hierarchy = {}
    label_hierarchy['Root'] = []
    for line in train_data:
        line = line.rstrip('\n')
        line = json.loads(line)
        line = line['doc_label']
        if line[0] in label_hierarchy:
            if line[1] not in label_hierarchy[line[0]]:
                label_hierarchy[line[0]].append(line[1])
                if line[2] != 'unknown':
                    label_hierarchy[line[1]]=[line[2]]
            else:
                if line[2] != 'unknown':
                    if line[1] in label_hierarchy:
                        if line[2] not in label_hierarchy[line[1]]:
                            label_hierarchy[line[1]].append(line[2])
                    else:
                        label_hierarchy[line[1]]=[line[2]]
        else:
            label_hierarchy['Root'].append(line[0])
            label_hierarchy[line[0]] = [line[1]]
            if line[2] != 'unknown':
                label_hierarchy[line[1]] = [line[2]]
    for line in test_data:
        line = line.rstrip('\n')
        line = json.loads(line)
        line = line['doc_label']
        if line[0] in label_hierarchy:
            if line[1] not in label_hierarchy[line[0]]:
                label_hierarchy[line[0]].append(line[1])
                if line[2] != 'unknown':
                    label_hierarchy[line[1]] = [line[2]]
            else:
                if line[2] != 'unknown':
                    if line[1] in label_hierarchy:
                        if line[2] not in label_hierarchy[line[1]]:
                            label_hierarchy[line[1]].append(line[2])
                    else:
                        label_hierarchy[line[1]] = [line[2]]
        else:
            label_hierarchy['Root'].append(line[0])
            label_hierarchy[line[0]] = [line[1]]
            if line[2] != 'unknown':
                label_hierarchy[line[1]] = [line[2]]
    f = open('amazon.taxonomy', 'w')
    for i in label_hierarchy.keys():
        line = [i]
        line.extend(label_hierarchy[i])
        line = '\t'.join(line) + '\n'
        f.write(line)
    f.close()

if __name__ == '__main__':
    get_data_from_meta('train_40k.csv')
    get_data_from_meta('val_10k.csv')
    get_hierarchy()
