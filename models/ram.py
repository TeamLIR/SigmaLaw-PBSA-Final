# -*- coding: utf-8 -*-
# file: ram.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.squeeze_embedding import SqueezeEmbedding


class RAM(nn.Module):
    def locationed_memory(self, memory, memory_len, left_len, aspect_len):
        batch_size = memory.shape[0]
        seq_len = memory.shape[1]
        memory_len = memory_len.cpu().numpy()
        left_len = left_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        u = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for idx in range(left_len[i]):
                weight[i].append(1 - (left_len[i] - idx) / memory_len[i])
                u[i].append(idx - left_len[i])
            for idx in range(left_len[i], left_len[i] + aspect_len[i]):
                weight[i].append(1)
                u[i].append(0)
            for idx in range(left_len[i] + aspect_len[i], memory_len[i]):
                weight[i].append(1 - (idx - left_len[i] - aspect_len[i] + 1) / memory_len[i])
                u[i].append(idx - left_len[i] - aspect_len[i] + 1)
            for idx in range(memory_len[i], seq_len):
                weight[i].append(1)
                u[i].append(0)
        u = torch.tensor(u, dtype=memory.dtype).to(self.opt.device).unsqueeze(2)
        weight = torch.tensor(weight).to(self.opt.device).unsqueeze(2)
        v = memory * weight
        memory = torch.cat([v, u], dim=2)
        return memory

    def __init__(self, bert, opt):
        super(RAM, self).__init__()
        self.opt = opt
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)

        self.bi_lstm_context = DynamicLSTM(opt.bert_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                           bidirectional=True)

        self.att_linear = nn.Linear(opt.hidden_dim * 2 + 1 + opt.bert_dim * 2, 1)

        self.gru_cell = nn.GRUCell(opt.hidden_dim * 2 + 1, opt.bert_dim)

        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, aspect_indices, text_left_indices = inputs[0], inputs[1], inputs[2]
        left_len = torch.sum(text_left_indices != 0, dim=-1)
        memory_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = aspect_len.float()

        # Embedding layer
        context = self.squeeze_embedding(text_raw_indices, memory_len)
        hid_context = self.bert(text_raw_indices)[2]
        context = torch.stack(hid_context[-4:]).sum(0)
        memory = self.dropout(context)
        memory, (_, _) = self.bi_lstm_context(memory, memory_len)

        memory = self.locationed_memory(memory, memory_len, left_len, aspect_len)

        aspect = self.squeeze_embedding(aspect_indices, aspect_len)
        hid_aspect = self.bert(aspect)[2]
        aspect = torch.stack(hid_aspect[-4:]).sum(0)
        aspect = self.dropout(aspect)

        # memory = self.embed(text_raw_indices)

        # aspect = self.embed(aspect_indices)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.unsqueeze(-1))
        et = torch.zeros_like(aspect).to(self.opt.device)

        batch_size = memory.size(0)
        seq_len = memory.size(1)
        memory = memory.to(torch.double)
        for _ in range(self.opt.hops):
            g = self.att_linear(torch.cat([memory.float(),
                                           torch.zeros(batch_size, seq_len, self.opt.bert_dim).to(
                                               self.opt.device) + et.unsqueeze(1),
                                           torch.zeros(batch_size, seq_len, self.opt.bert_dim).to(
                                               self.opt.device) + aspect.unsqueeze(1)],
                                          dim=-1))
            alpha = F.softmax(g, dim=1)
            i = torch.bmm(alpha.transpose(1, 2), memory.float()).squeeze(1)
            et = self.gru_cell(i, et)

        out = self.dense(et)
        return out
