# -*- coding: utf-8 -*-
# file: atae-lstm
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
from layers.attention import Attention, NoQueryAttention
from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn

from layers.squeeze_embedding import SqueezeEmbedding

import torch
import torch.nn as nn


class ATAE_LSTM(nn.Module):
    def __init__(self, bert, opt):
        super(ATAE_LSTM, self).__init__()
        self.opt = opt
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)

        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(opt.bert_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention = NoQueryAttention(opt.hidden_dim + opt.bert_dim, score_function='bi_linear')
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.as_tensor(torch.sum(aspect_indices != 0, dim=-1), dtype=torch.float).to(self.opt.device)

        context = self.squeeze_embedding(text_raw_indices, x_len)
        context = self.bert(context)[0]
        context = self.dropout(context)

        aspect = self.squeeze_embedding(aspect_indices, aspect_len)
        aspect = self.bert(aspect)[0]
        aspect = self.dropout(aspect)

        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.view(aspect_len.size(0), 1))
        aspect = torch.unsqueeze(aspect_pool, dim=1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, context), dim=-1)

        h, (_, _) = self.lstm(context, x_len)
        ha = torch.cat((h, aspect), dim=-1)
        _, score = self.attention(ha)
        output = torch.squeeze(torch.bmm(score, h), dim=1)

        out = self.dense(output)
        return out
