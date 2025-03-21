# -*- coding: utf-8 -*-
# coding=utf-8
# coding:utf-8
import sys
from pdb import set_trace as stop


sys.path.append('../')
import torch
import numpy as np
import torch.nn as nn
from transformers import BertModel, BertConfig
import torch.nn.functional as F
from torch import nn

bert = BertModel.from_pretrained("../bert-chinese")
# bert = BertModel.from_pretrained('bert-base-uncased')

class PT_BERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(768, 8)  # 分类器t
        self.Bert_encoder = bert#Bert模型

    def forward(self, inputs):
        text_bert_sequence= inputs[0]
        bert_out, _ = self.Bert_encoder(text_bert_sequence,return_dict=False)
        logits = self.classifier(bert_out.sum(1, keepdim=True).squeeze(1))
        return logits,bert_out.sum(1, keepdim=True).squeeze(1)

class SupConLoss(nn.Module):
    """https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # print("features.size():",features.size())
        # print(labels)
        # print("labels.size():",labels.size())

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            # print("tag:", 1)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            # print(labels)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            # print("tag:", 2)
        else:
            mask = mask.float().to(device)
        #     print("tag:", 3)
        #
        # print("mask:", mask)
        # print("mask.size():", mask.size())

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # print("contrast_feature:", contrast_feature)
        # print("contrast_feature.size():", contrast_feature.size())

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # print("anchor_feature:", anchor_feature)
        # print("anchor_feature.size():", anchor_feature.size())

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # print("anchor_dot_contrast:", anchor_dot_contrast)
        # print("anchor_dot_contrast.size():", anchor_dot_contrast.size())

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # print("logits:", logits)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # print("mask:", mask)
        # print("mask.size():", mask.size())

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # print("logits_mask:", logits_mask)
        # print("logits_mask.size():", logits_mask.size())

        mask = mask * logits_mask

        # print("mask:", mask)
        # print("mask.size():", mask.size())

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-30)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        # print("loss:", loss)
        # print("loss.size():", loss.size())

        return loss
