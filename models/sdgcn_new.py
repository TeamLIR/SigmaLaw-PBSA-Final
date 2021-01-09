
import torch
import torch.nn as nn

from layers.dynamic_rnn import DynamicLSTM
from layers.slstm import sLSTM
from layers.attention import Attention
from layers.squeeze_embedding import SqueezeEmbedding


class SDGCN_NEW(nn.Module):
    def __init__(self, bert, opt):
        super(SDGCN_NEW, self).__init__()
        self.opt = opt
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)

        self.lstm_context = sLSTM(opt.bert_dim, 768,
                                  window_size=2)
        self.lstm_aspect = sLSTM(opt.bert_dim, 768,
                                 window_size=2)
        # self.lstm_context = DynamicLSTM(opt.bert_dim, opt.hidden_dim, num_layers=1, batch_first=True,bidirectional=True)
        # self.lstm_aspect = DynamicLSTM(opt.bert_dim, opt.hidden_dim, num_layers=1, batch_first=True,bidirectional=True)

        self.attention_aspect = Attention(768, score_function='bi_linear')
        self.attention_context = Attention(768, score_function='bi_linear')
        self.dense = nn.Linear(768 *2, opt.polarities_dim)


    def forward(self, inputs):
        context, aspect, sen_lens,aspect_lens,tree = inputs[0], inputs[1], inputs[2],inputs[3],inputs[4]
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

        context = context.permute(1 ,0 ,2)

        aspect = aspect.permute(1 ,0 ,2)

        sen_lens = torch.unsqueeze(sen_lens ,dim=1)
        aspect_lens = torch.unsqueeze(aspect_lens ,dim=1)
        # context, (_, _) = self.lstm_context(context, context_len)
        # aspect, (_, _) = self.lstm_aspect(aspect, aspect_len)
        print("context: ", context.size())
        print("sen_lens: ", sen_lens.size())

        _, context = self.lstm_context((context, sen_lens))
        _, aspect= self.lstm_aspect((aspect, aspect_lens))
        context = context.permute(1, 0, 2)

        aspect = aspect.permute(1, 0, 2)
        aspect_len = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)
        aspect_pool = torch.sum(aspect, dim=1)
        aspect_pool = torch.div(aspect_pool, aspect_len.view(aspect_len.size(0), 1))

        text_raw_len = torch.tensor(context_len, dtype=torch.float).to(self.opt.device)
        context_pool = torch.sum(context, dim=1)
        context_pool = torch.div(context_pool, text_raw_len.view(text_raw_len.size(0), 1))

        print("aspect", aspect.size())
        print("context_pool", context_pool.size())
        aspect_final, _ = self.attention_aspect(aspect, context_pool)
        aspect_final = aspect_final.squeeze(dim=1)
        context_final, _ = self.attention_context(context, aspect_pool)
        context_final = context_final.squeeze(dim=1)

        x = torch.cat((aspect_final, context_final), dim=-1)
        out = self.dense(x)
        return out
