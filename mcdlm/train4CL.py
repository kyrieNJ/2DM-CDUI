# -*- coding: utf-8 -*-
# coding=utf-8
# encoding=utf-8
import sys
from torch.utils.data import DataLoader
from model4cd import PT_BERTEncoder, SupConLoss
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import random
import math
import argparse
import numpy as np
import torch.nn as nn
from transformers import BertModel

from sklearn import metrics
from vocab import Vocab
from data_utils4CL import CLDataset
from tqdm import tqdm
import pickle
import torch.utils.data.distributed

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

class Instructor:
    def __init__(self, args):
        self.args = args

        #数据加载阶段
        self.trainset = CLDataset(args.dataset_file['train'],args)
        #模型训练阶段
        self.model = args.model_class()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.model.load_state_dict(torch.load(self.args.state_dict_path))
        self.model.to(args.device)
        self._print_args()
        #****
        bert_model = self.model.Bert_encoder
        bert_params_dict = list(map(id, bert_model.parameters()))
        base_params = filter(lambda p: id(p) not in bert_params_dict, self.model.parameters())
        optimizer_grouped_parameters = [
            {"params": base_params},
            {"params": bert_model.parameters(), "lr": args.bert_lr},
        ]
        # trick2
        # optimizer_grouped_parameters=get_parameters(bert_model,args.bert_lr,1)

        self.optimizer = self.args.optimizer(
            optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.l2reg
        )
        # self.model=torch.nn.DataParallel(self.model,device_ids=device_ids)
        # self.model= self.model.cuda(device=0)
        self.global_acc = 0.
        self.global_im = 0.

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=args.device.index))

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

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.args.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, optimizer):
        global_step = 0
        continue_not_increase = 0
        similar_criterion = SupConLoss()
        for epoch in range(self.args.num_epoch):
            print('-' * 100)
            print('epoch: ', epoch)
            n_correct, n_total,loss_total = 0, 0,0
            increase_flag = False
            self.model.train()  # 训练过程
            for i_batch, sample_batched in tqdm(enumerate(self.train_data_loader)):
            # for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.args.device) for col in self.args.inputs_cols]

                _,outputs= self.model(inputs)
                targets = sample_batched['label'].to(self.args.device)
                loss = similar_criterion(outputs.unsqueeze(1), labels=torch.tensor(targets))

                # loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)

                if global_step % self.args.log_step == 0:
                    train_loss = loss_total / n_total
                    # print('train_acc:',train_acc)
                    print("train loss: {:.4f} ".format(train_loss))
                    #       "valid explicit acc: {:.4f}, valid implicit acc: {:.4f} ".format(explicit_acc, implicit_acc))

                if global_step % self.args.save_frequency == 0 and global_step != 0 :
                    model_file = "CDCL_min_epoch_{}_step_{}.pkl".format(epoch, global_step)
                    # torch.save(self.model.state_dict(), 'state_dict/' + self.args.model_name +'_'+ self.args.dataset + '.pkl')
                    torch.save(self.model.state_dict(), 'state_dict/' + model_file)
                    print("Model saved: {}".format(model_file))


        return train_loss


    def run(self):
        # Loss and Optimizer
        # _params = filter(lambda p: p.requires_grad, self.model.parameters())
        # optimizer = self.args.optimizer(_params, lr=self.args.learning_rate, weight_decay=self.args.l2reg)

        self.train_data_loader = DataLoader(dataset=self.trainset, batch_size=args.batch_size, shuffle=False,num_workers=2,drop_last=True)
        if not os.path.exists('log/'):
            os.mkdir('log/')

        f_out = open('log/' + self.args.model_name + '_' + self.args.dataset + '_CDCL.txt', 'w', encoding='utf-8')

        for i in range(args.repeat):
            print('repeat: ', (i+1))
            f_out.writelines('repeat: '+str(i+1))
            self._reset_params()
            train_loss = self._train(self.optimizer)
            print('train_loss: {0}         '.format(train_loss))
            f_out.writelines('train_loss: {0}\n'.format(train_loss))
            print('-' * 100)

        f_out.close()

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

if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_dict_path", type=str, default="state_dict/ori_PT_BERTEncoder_CD.pkl")
    parser.add_argument("--save_frequency", type=str, default=280440)

    parser.add_argument('--model_name', default='PT_BERTEncoder', type=str)
    parser.add_argument('--dataset', default='CL', type=str)

    #orthogonal_，xavier_uniform_
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument("--learning_rate", type=float, default=1e-7, help="learning rate.")
    parser.add_argument("--bert_lr", type=float, default=2e-6, help="learning rate for bert.")
    parser.add_argument("--l2reg", type=float, default=1e-5, help="weight decay rate.")
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument("--num_epoch", type=int, default=50, help="Number of total training epochs.")

    parser.add_argument("--batch_size", type=int, default=9, help="Training batch size.")
    parser.add_argument("--log_step", type=int, default=20, help="Print log every k steps.")

    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument("--max_len", type=int, default=130)

    parser.add_argument('--repeat', default=1, type =int)

    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument("--lower", default=True, help="Lowercase all words.")
    parser.add_argument("--direct", default=False)
    parser.add_argument("--loop", default=True)
    parser.add_argument("--reset_pooling", default=False, action="store_true")
    parser.add_argument("--output_merge",type=str,default="none",help="merge method to use, (none, gate)",)
    args = parser.parse_args()


    model_classes = {
        'PT_BERTEncoder':PT_BERTEncoder,

    }
    input_colses = {
        'PT_BERTEncoder': [
            'bert_text_sequence',
        ],
    }

    dataset_files = {
        'CL': {
            'train': '../datasets/CD/CL_dataset_8min.json',
        },
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamw': torch.optim.AdamW,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    args.model_class = model_classes[args.model_name]
    args.inputs_cols = input_colses[args.model_name]
    args.dataset_file = dataset_files[args.dataset]
    args.initializer = initializers[args.initializer]
    args.optimizer = optimizers[args.optimizer]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device is None else torch.device(args.device)
    args.torch_version = torch.__version__

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    ins = Instructor(args)
    ins.run()

