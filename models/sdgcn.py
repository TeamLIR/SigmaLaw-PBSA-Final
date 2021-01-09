import torch
import torch.nn as nn

from layers.attention import Attention
from layers.dynamic_rnn import DynamicLSTM
from layers.squeeze_embedding import SqueezeEmbedding


class SDGCN(nn.Module):
    def __init__(self, bert, opt):
        super(SDGCN, self).__init__()
        self.opt = opt
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)

        self.lstm_context = DynamicLSTM(opt.bert_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_aspect = DynamicLSTM(opt.bert_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention_aspect = Attention(opt.hidden_dim, score_function='bi_linear')
        self.attention_context = Attention(opt.hidden_dim, score_function='bi_linear')
        self.dense = nn.Linear(opt.hidden_dim*2, opt.polarities_dim)


    def forward(self, inputs):
        context, aspect = inputs[0], inputs[1]
        context_len = torch.sum(context != 0, dim=-1)
        aspect_len = torch.sum(aspect != 0, dim=-1)

        # Embedding layer

        context = self.squeeze_embedding(context, context_len)
        hid_context = self.bert(context)[2]
        context= torch.stack(hid_context[-4:]).sum(0)
        context = self.dropout(context)

        aspect = self.squeeze_embedding(aspect, aspect_len)
        hid_aspect = self.bert(aspect)[2]
        aspect= torch.stack(hid_aspect[-4:]).sum(0)
        aspect = self.dropout(aspect)

        # LSTM Layer

        context, (_, _) = self.lstm_context(context, context_len)
        aspect, (_, _) = self.lstm_aspect(aspect, aspect_len)

        # Attention Layer

        aspect_len = torch.as_tensor(aspect_len, dtype=torch.float).to(self.opt.device)
        aspect_pool = torch.sum(aspect, dim=1)
        aspect_pool = torch.div(aspect_pool, aspect_len.view(aspect_len.size(0), 1))

        text_raw_len = torch.as_tensor(context_len, dtype=torch.float).to(self.opt.device)
        context_pool = torch.sum(context, dim=1)
        context_pool = torch.div(context_pool, text_raw_len.view(text_raw_len.size(0), 1))

        aspect_final, _ = self.attention_aspect(aspect, context_pool)
        aspect_final = aspect_final.squeeze(dim=1)
        context_final, _ = self.attention_context(context, aspect_pool)
        context_final = context_final.squeeze(dim=1)

        x = torch.cat((aspect_final, context_final), dim=-1)

        # Output Layer
        out = self.dense(x)
        return out
