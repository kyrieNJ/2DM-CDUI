# -*- coding: utf-8 -*-
# coding=utf-8
# encoding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from torch.utils.data import DataLoader
from model4cd import PT_BERTEncoder, SupConLoss
import torch
import random
import math
from tqdm import tqdm
import spacy
import argparse
from vocab import Vocab
import numpy as np

import pickle
import torch.nn.functional as F
from transformers import BertTokenizer
import json


nlp = spacy.load('zh_core_web_sm')

# device = torch.device(device=2)

tokenizer = BertTokenizer.from_pretrained('../bert-chinese')



#--------------------------------data-utlis-bert---------------------------------
def parse_json_all(path):
    with open(path, 'r',encoding='utf-8-sig') as file:
        data = json.load(file)
    tmpdict=data
    return tmpdict

def text_to_bert_sequence(text, max_len,padding="post", truncating="post"):
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


#---------------------------------bert-Infer--------------------------------

def get_parameters(model, model_init_lr, multiplier):
    parameters = []
    enc_param_optimizer = list(model.named_parameters())
    lr = model_init_lr
    for layer in range(12, -1, -1):
        layer_params = {
            'params': [p for n, p in enc_param_optimizer if f'encoder.layer.{layer}.' in n],
            'lr': lr,
            'weight_decay': 0.0
        }
        parameters.append(layer_params)
        lr *= multiplier
    return parameters

class Inferer:
    def __init__(self, args):
        self.args = args

        #模型加载阶段ASGCN,DKF-BERT
        self.model = args.model_class()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.model.to(args.device)
        self.model.load_state_dict(torch.load(self.args.state_dict_path))
        self._print_args()
        #****
        bert_model = self.model.Bert_encoder
        bert_params_dict = list(map(id, bert_model.parameters()))
        base_params = filter(lambda p: id(p) not in bert_params_dict, self.model.parameters())
        optimizer_grouped_parameters = [
            {"params": base_params},
            {"params": bert_model.parameters(), "lr": args.bert_lr},
        ]

        self.optimizer = torch.optim.Adam(
            optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.l2reg
        )

        self.global_f1 = 0.
        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=args.device.index))

        self.model = self.model
        self.model.to(args.device)
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.args):
            print('>>> {0}: {1}'.format(arg, getattr(self.args, arg)))


    def processtext(self, user_data_dict):
        user_count = 0
        all_data = []
        for iuser in tqdm(user_data_dict):
            twi_text_bertlist = []
            if len(iuser['tweets']) > 100:
                iuser['tweets'] = iuser['tweets'][:100]
            for itw in iuser['tweets']:
                twi_text = itw['tweet_content'].strip()

                bert_text_sequence = text_to_bert_sequence(twi_text,75)
                twi_text_bertlist.append(bert_text_sequence)

            list_pad = [0] * (75)
            # nodeist_pad=[0] * (args.sm_max_len)
            list_pad = np.array(list_pad)
            for i in range(0, 100 - len(twi_text_bertlist)):
                twi_text_bertlist.append(list_pad)

            data = {
                'twi_text_bert_sequence_list': np.array(twi_text_bertlist),
            }
            all_data.append(data)
            user_count += 1
        return all_data


    def evaluate(self, user_data_dict):
        pdata=self.processtext(user_data_dict)
        count=0
        cd_dict={}
        pdata1 = DataLoader(dataset=pdata, batch_size=args.batch_size, shuffle=False,num_workers=2)
        for i_batch, sample_batched in tqdm(enumerate(pdata1)):
            inputs = [sample_batched['twi_text_bert_sequence_list'].to(self.args.device) ]
            tensor_inputs=inputs[0].squeeze(0)
            tmplist=[]
            tmplist.append(tensor_inputs)
            _,outputs = self.model(tmplist)
            cd_dict[count]=outputs
            count+=1
        return cd_dict


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    #ori_CL_min_epoch_49_step_701100,CL_PT_BERTEncoder_CD.pkl,CDCL_min_epoch_39_step_560880.pkl,ori_PT_BERTEncoder_CD.pkl
    parser.add_argument("--state_dict_path", type=str, default="state_dict/CDCL_min_epoch_39_step_560880.pkl")

    parser.add_argument('--model_name', default='PT_BERTEncoder', type=str)
    parser.add_argument('--dataset', default='CL', type=str)

    # orthogonal_，xavier_uniform_
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument("--learning_rate", type=float, default=1e-7, help="learning rate.")
    parser.add_argument("--bert_lr", type=float, default=2e-6, help="learning rate for bert.")
    parser.add_argument("--l2reg", type=float, default=1e-5, help="weight decay rate.")
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument("--num_epoch", type=int, default=30, help="Number of total training epochs.")

    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--log_step", type=int, default=20, help="Print log every k steps.")

    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument("--max_len", type=int, default=100)

    parser.add_argument('--repeat', default=1, type=int)

    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument("--lower", default=True, help="Lowercase all words.")
    parser.add_argument("--direct", default=False)
    parser.add_argument("--loop", default=True)
    parser.add_argument("--reset_pooling", default=False, action="store_true")
    parser.add_argument("--output_merge", type=str, default="none", help="merge method to use, (none, gate)", )
    args = parser.parse_args()

    model_classes = {
        'PT_BERTEncoder': PT_BERTEncoder,

    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # defa ult lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamw': torch.optim.AdamW,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    args.model_class = model_classes[args.model_name]
    args.initializer = initializers[args.initializer]
    args.optimizer = optimizers[args.optimizer]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device is None else torch.device(args.device)
    # args.device = torch.device("cuda:0")
    args.torch_version = torch.__version__

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # fname=  {
    #     'train': '../datasets/SWDD/SWDD_1k8_train.json',
    #     'test': '../datasets/SWDD/SWDD_1k8_test.json'
    # }
    fname=  {
        'train': '../datasets/SWDD/SWDD_3k7_train.json',
        'test': '../datasets/SWDD/SWDD_3k7_test.json'
    }
    user_data_dict = parse_json_all(fname['train'])
    ins = Inferer(args)
    cd_dict = ins.evaluate(user_data_dict)
    print(len(cd_dict))
    foutds = open('./SWDD_3k7_CDCL_twitext_embedding_train_100.pkl', 'wb')
    pickle.dump(cd_dict, foutds)
    foutds.close()

    user_data_dict = parse_json_all(fname['test'])
    ins = Inferer(args)
    cd_dict = ins.evaluate(user_data_dict)
    print(len(cd_dict))
    foutds = open('./SWDD_3k7_CDCL_twitext_embedding_test_100.pkl', 'wb')
    pickle.dump(cd_dict, foutds)
    foutds.close()

