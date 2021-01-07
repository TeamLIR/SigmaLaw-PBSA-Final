

import torch
import torch.nn as nn

from layers.bilinear_attention import Bilinear_Attention
from layers.dynamic_rnn import DynamicLSTM
from layers.squeeze_embedding import SqueezeEmbedding
from layers.attention import Attention, NoQueryAttention


import torch
import torch.nn as nn


class SDGCN(nn.Module):
    def __init__(self, bert, opt):
        super(SDGCN, self).__init__()
        self.opt = opt
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)

        self.lstm_context = DynamicLSTM(opt.bert_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_aspect = DynamicLSTM(opt.bert_dim, opt.hidden_dim, num_layers=1, batch_first=True)

        self.attention_aspect = Bilinear_Attention(opt.hidden_dim )
        self.attention_context = Bilinear_Attention(opt.hidden_dim )

        self.dense = nn.Linear(opt.hidden_dim * 2, opt.polarities_dim)

    def forward(self, inputs):
        context, aspect, all_aspects, aspects_num_max, all_targets_position, targets_all_len = inputs[0], inputs[1], inputs[2], self.opt.aspect_num_max, inputs[3], inputs[4]
        context_len = torch.sum(context != 0, dim=-1)
        aspect_len = torch.sum(aspect != 0, dim=-1)
        all_aspects_max_len = torch.sum(all_aspects != 0, dim=-1)

        # print("context: " , context.size())
        # print("aspect: " , aspect.size())
        # print("all_aspects: " , all_aspects.size())
        # print("context_len: " , context_len.size())
        # print("aspect_len: " , aspect_len.size())
        # print("all_aspects_max_len: " , all_aspects_max_len.size())

        #batch_size = inputs.size()[0]
        targets_all_len = []
        for i in range(aspects_num_max):
            #targets_i_len = tf.slice(self.targets_all_len_a, [0, i], [batch_size, 1])
            targets_i_len = targets_all_len[:,i,:]
            targets_all_len.append(torch.squeeze(targets_i_len))

            # position embedding
        targets_all_position = []
        for i in range(aspects_num_max):
            targets_i_pos = all_targets_position[:, i, :]
            targets_all_position.append(torch.squeeze(targets_i_pos))
            # print("targets_i_len: " , targets_i_len.size())

        # Embedding layer

        # Embedding for the context
        context = self.squeeze_embedding(context, context_len)
        context = self.bert(context)[0]
        context = self.dropout(context)
        # print("context: " , context.size())

        # Embedding for the aspect
        aspect = self.squeeze_embedding(aspect, aspect_len)
        aspect = self.bert(aspect)[0]
        aspect = self.dropout(aspect)
        # print("aspect: " , aspect.size())

        # Embedding for the all targets
        embedded_aspects_all = list(range(aspects_num_max))
        for i in range(aspects_num_max):
            # get a target
            aspect_i = all_aspects[:, i, :]
            #  aspect_i = self.squeeze_embedding(aspect_i, all_aspects_max_len)
            aspect_i = self.bert(aspect_i)[0]
            aspect_i = self.dropout(aspect_i)
            embedded_aspects_all[i] = aspect_i

        # LSTM Layer

        # Bi-LSTM for the context
        context, (_, _) = self.lstm_context(context, context_len)
        # print("context: " , context.size())
        text_raw_len = torch.tensor(context_len, dtype=torch.float).to(self.opt.device)
        # print("text_raw_len: " , text_raw_len.size())
        context_pool = torch.sum(context, dim=1)
        context_pool = torch.div(context_pool, text_raw_len.view(text_raw_len.size(0), 1))
        # print("context_pool: " , context_pool.size())

        # Bi-LSTM for the aspects
        LSTM_aspects_all = list(range(aspects_num_max))

        pool_aspects_all = list(range(aspects_num_max))

        aspect_len_max = torch.tensor(all_aspects_max_len, dtype=torch.float).to(self.opt.device)

        for i in range(aspects_num_max):
            LSTM_aspects_all[i], (_, _) = self.lstm_aspect(embedded_aspects_all[i], aspect_len)
            # print("LSTM_aspects_all[i]: " , LSTM_aspects_all[i].size())
            # pool_aspects_i = torch.sum(LSTM_aspects_all[i], dim=1)
            # pool_aspects_i = torch.div(pool_aspects_i, aspect_len_max.view(aspect_len_max.size(0), 1))
            # print("pool_aspects_i: " , pool_aspects_i.size())

        # Attention layer
        outputs_ss = list(range(aspects_num_max))  # all the target attention for the sentence
        outputs_ts = list(range(aspects_num_max))
        for i in range(aspects_num_max):
            print("inputs: ", LSTM_aspects_all[i].size(), context_pool.size(), targets_all_len[i].size())
            att_s_i, _ = self.attention_aspect(LSTM_aspects_all[i], context_pool, targets_all_len[i])
            outputs_ss[i] = torch.squeeze(torch.matmul(att_s_i, LSTM_aspects_all[i]), axis=1)  # 13*(?,600)

            # position
            target_position_i = torch.unsqueeze(targets_all_position[i], 2)  # (?,78,1)
            LSTM_Hiddens_sen_i = torch.matmul(context_pool, target_position_i)

            att_s_i = self.attention_aspect(LSTM_Hiddens_sen_i, outputs_ss[i], context_len)
            outputs_ts[i] = torch.squeeze(torch.matmul(att_s_i, LSTM_Hiddens_sen_i), axis=1)

        x = torch.cat((torch.unsqueeze(i, axis=2) for i in self.outputs_ts), dim=-1)
        out = self.dense(x)
        return out