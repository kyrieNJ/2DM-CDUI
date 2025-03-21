# -*- coding: utf-8 -*-
#!/usr/bin/env python
# coding=utf-8
import os
import pickle

import spacy
from vocab import Vocab
import numpy as np
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer
from tqdm import tqdm
import json

tokenizer = BertTokenizer.from_pretrained('../bert-chinese')


def parse_json_all(path):
    with open(path, 'r',encoding='utf-8-sig') as file:
        data = json.load(file)
    return data

def text_to_bert_sequence(text, max_len, maxtextlen,padding="post", truncating="post"):
    text = tokenizer.tokenize(text)
    text = ["[CLS]"] + text + ["[SEP]"]
    sequence = tokenizer.convert_tokens_to_ids(text)
    if maxtextlen < len(text)-2:
        maxtextlen = len(text)-2
        print(maxtextlen)
    return pad_and_truncate(sequence, max_len, padding=padding, truncating=truncating),maxtextlen

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc

    return x

class CLDataset(Dataset):
    def __init__(self, fname,args):
        cl_data_dict = parse_json_all(fname)
        fin = open('./CL_tmptext_BERTtoken.pkl', 'rb')
        token_dict = pickle.load(fin)
        fin.close()
        all_data = []
        maxcount=0
        maxtwilen=0
        for ikey in tqdm(cl_data_dict.keys()):
            for icllist in cl_data_dict[ikey]:
                lable = [1,0, 0, 0, 1, 0, 0, 0,0]
                lacount=0
                for itext in icllist:
                    # bert_text_sequence,maxtextlen = text_to_bert_sequence(itext, args.max_len,maxtextlen)
                    bert_text_sequence=token_dict[itext]
                    data = {
                        'bert_text_sequence':bert_text_sequence,
                        'label': lable[lacount],
                    }
                    all_data.append(data)
                    lacount+=1
                    maxcount+=1
        print('maxtwilen:', maxtwilen)
        print('maxcount:', maxcount)
        self.data = all_data


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)