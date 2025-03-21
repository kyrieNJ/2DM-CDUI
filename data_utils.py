# -*- coding: utf-8 -*-
#!/usr/bin/env python
# coding=utf-8
import os
import pickle
from queue import Queue

import spacy

from vocab import Vocab
import numpy as np
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer
from tqdm import tqdm
import json

tokenizer = BertTokenizer.from_pretrained('./bert-chinese')

nlp = spacy.load('zh_core_web_sm')

def parse_json_all(path):
    with open(path, 'r',encoding='utf-8-sig') as file:
        data = json.load(file)
    tmpdict=data
    return tmpdict

def text_to_bert_sequence(text, max_len, padding="post", truncating="post"):
    text = tokenizer.tokenize(text)
    text = ["[CLS]"] + text + ["[SEP]"]
    sequence = tokenizer.convert_tokens_to_ids(text)
    return pad_and_truncate(sequence, max_len, padding=padding, truncating=truncating)


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


class MM_PDUDDataset(Dataset):
    def __init__(self, fname,args):
        foutPIaBI = open(fname + '_PIaBI_dict.pkl', 'rb')
        u_PIaBI_dic = pickle.load(foutPIaBI)
        foutPIaBI.close()
        #
        if 'train' in fname:
            fin =  open('./pretrain/SWDD_oriCL_twitext_embedding_train_100.pkl', 'rb')
        else:
            fin =  open('./pretrain/SWDD_oriCL_twitext_embedding_test_100.pkl', 'rb')
        cd_dict = pickle.load(fin)
        fin.close()


        user_data_dict = parse_json_all(fname)
        maxtwilen=0
        user_count=0
        all_data = []

        for iuser in tqdm(user_data_dict):
            twi_text_bertlist=[]
            if len(iuser['tweets']) > 100:
                iuser['tweets'] = iuser['tweets'][:100]
            user_BI=[]
            twi_count=0
            for itw in iuser['tweets']:
                twi_text = itw['tweet_content'].strip()
                bert_text_sequence = text_to_bert_sequence(twi_text, args.max_len)
                twi_text_bertlist.append(bert_text_sequence)

                twi_count+=1


            twi_cd_emblist=cd_dict[user_count]


            ulable=iuser['dep_label']

            list_pad = [0] * (args.max_len)
            list_pad = np.array(list_pad)
            for i in range(0,args.twi_max_len - len(twi_text_bertlist)):
                twi_text_bertlist.append(list_pad)

            #PaB
            PI_list=u_PIaBI_dic[user_count]['PI']
            sex_bert_sequence = np.array(text_to_bert_sequence(PI_list[0], 30))

            BI_list=u_PIaBI_dic[user_count]['BI']
            ori_bert_sequence = np.array(text_to_bert_sequence(BI_list[0], 30))
            incI_bert_sequence = np.array(text_to_bert_sequence(BI_list[1], 30))
            laN_bert_sequence = np.array(text_to_bert_sequence(BI_list[2], 30))

            for ibitext in BI_list:
                bert_BI_sequence = np.array(text_to_bert_sequence(ibitext, 30))
                user_BI.append(bert_BI_sequence)

            data = {
                'twi_text_bert_sequence_list':np.array(twi_text_bertlist),
                'twi_cd_list': twi_cd_emblist,
                'sex_bert_sequence':sex_bert_sequence,
                'ori_bert_sequence':ori_bert_sequence,
                'incI_bert_sequence': incI_bert_sequence,
                'laN_bert_sequence': laN_bert_sequence,
                'risk_label': int(ulable),
            }
            all_data.append(data)
            user_count+=1
        print('maxtwilen:', maxtwilen)
        self.data = all_data


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)