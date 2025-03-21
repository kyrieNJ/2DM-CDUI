# -*- coding: utf-8 -*-
# coding=utf-8
# coding:utf-8
import os
import sys
from models.CRA import Ranked_Attention as RA
from models.model_utils import InteractiveAttention
from models.multiheadattention import MultiHeadedAttention as MA

sys.path.append('../')
import torch
from transformers import BertModel
import torch.nn.functional as F
from torch import nn


bert = BertModel.from_pretrained("./bert-chinese")


class MM_Encoder(nn.Module):
    def __init__(self, args):
    # def __init__(self, args):

        super().__init__()
        self.args = args
        # self.classifier = nn.Linear(args.bert_out_dim , args.num_class)  # 分类器t
        # wo MAWM
        self.classifier = nn.Linear(args.bert_out_dim, args.num_class)  # 分类器t

        self.Bert_encoder = bert#Bert模型

        self.in_drop = nn.Dropout(args.bert_dropout)  # Twitter  bert输出时第一次dropout

        self.RA = RA()

        self.fc1=nn.Linear(args.bert_out_dim*2, args.bert_out_dim)

        self.fc2 = nn.Linear(args.hidden_dim, args.bert_out_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.bert_out_dim)
        self.fc4 = nn.Linear(args.hidden_dim, args.bert_out_dim)
        self.fc5 = nn.Linear(args.hidden_dim, args.bert_out_dim)

        self.IAttention = InteractiveAttention(dropout=args.dropout).to(args.device)


    def forward(self, inputs):
        twi_text_bert_sequence_list,\
        twi_cd_emblist,\
        sex_bert_sequence,\
        ori_bert_sequence,\
        incI_bert_sequence,\
        laN_bert_sequence,\
            = inputs

        #twi-level
        batch_num=twi_text_bert_sequence_list.shape[0]
        all_twi_re =torch.randn(1,1,1)
        catflag=1

        for ibatch in range(0,batch_num):
            twi_text_out, _ = self.Bert_encoder(twi_text_bert_sequence_list[ibatch:ibatch+1,:,:].squeeze(0),return_dict=False)  # bert twit输出
            twi_text_out = self.in_drop(twi_text_out)  # bert输出时第一次dropout

            allasbrep1 = twi_text_out.mean(1).unsqueeze(0) #

            if catflag==1:
                all_twi_re=allasbrep1
                catflag=0
            else:
                all_twi_re=torch.cat((all_twi_re, allasbrep1), 0) #bx160x768

        #CD
        cdrep = twi_cd_emblist #bx160x768
        cdrep = self.in_drop(cdrep) #bx160x768


        outT,outCD=self.RA(all_twi_re,cdrep)

        fin_twi_rep = torch.cat((outT, outCD), 2) #bx160x600
        fin_twi_rep=self.fc1(fin_twi_rep).mean(1).unsqueeze(1)

        sex_out, _ = self.Bert_encoder(sex_bert_sequence, return_dict=False)  # bx30x768
        sex_out = self.fc2(sex_out).mean(1).unsqueeze(1)
        sex_out = self.in_drop(sex_out) #bx160x768

        ori_out, _ = self.Bert_encoder(ori_bert_sequence, return_dict=False)  # bx30x768
        ori_out = self.fc3(ori_out).mean(1).unsqueeze(1)
        ori_out = self.in_drop(ori_out) #bx160x768

        incI_out, _ = self.Bert_encoder(incI_bert_sequence, return_dict=False)  # bx30x768
        incI_out = self.fc4(incI_out).mean(1).unsqueeze(1)
        incI_out = self.in_drop(incI_out) #bx160x768
        #
        laN_out, _ = self.Bert_encoder(laN_bert_sequence, return_dict=False)  # bx30x768
        laN_out = self.fc5(laN_out).mean(1).unsqueeze(1)
        laN_out = self.in_drop(laN_out) #bx160x768

        fin_rep=torch.cat((fin_twi_rep,sex_out),1)
        fin_rep=torch.cat((fin_rep,ori_out),1)
        fin_rep=torch.cat((fin_rep,incI_out),1)
        fin_rep=torch.cat((fin_rep,laN_out),1)

        cat_outputs=self.IAttention(fin_rep, fin_rep).mean(1)

        logits = self.classifier(cat_outputs)

        return logits  # 2分类向量
