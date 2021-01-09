from layers.bilinear_attention import Bilinear_Attention
from layers.dynamic_rnn import DynamicLSTM
from layers.squeeze_embedding import SqueezeEmbedding
from layers.attention import Attention, NoQueryAttention


import torch
import torch.nn as nn




import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class SDGCN(nn.Module):
    def __init__(self, bert, opt):
        super(SDGCN, self).__init__()
        self.opt = opt
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)

        self.lstm_context = DynamicLSTM(opt.bert_dim, opt.hidden_dim, bidirectional=True,num_layers=1, batch_first=True)
        # self.lstm_aspect = DynamicLSTM(opt.bert_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        # self.attention_aspect = Attention(opt.hidden_dim, score_function='bi_linear')
        # self.attention_context = Attention(opt.hidden_dim, score_function='bi_linear')

        self.gc1 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc2 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.fc = nn.Linear(2 * opt.hidden_dim, opt.polarities_dim)
        #self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(1 - (aspect_double_idx[i, 0] - j) / context_len)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
                weight[i].append(1 - (j - aspect_double_idx[i, 1]) / context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).float().unsqueeze(2).to(self.opt.device)
        return weight * x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask * x


    def forward(self, inputs):
        context, aspect,adj,left_indices = inputs[0], inputs[1],inputs[2],inputs[3]
        context_len = torch.sum(context != 0, dim=-1)
        aspect_len = torch.sum(aspect != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)

        # Embedding layer

        # context = self.squeeze_embedding(context, context_len)
        hid_context = self.bert(context)[2]
        context= torch.stack(hid_context[-4:]).sum(0)
        context = self.dropout(context)

        # aspect = self.squeeze_embedding(aspect, aspect_len)
        # hid_aspect = self.bert(aspect)[2]
        # aspect= torch.stack(hid_aspect[-4:]).sum(0)
        # aspect = self.dropout(aspect)

        # LSTM Layer

        context, (_, _) = self.lstm_context(context, torch.tensor([90]))
        # aspect, (_, _) = self.lstm_aspect(aspect, aspect_len)

        # Attention Layer

        # aspect_len = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)
        # aspect_pool = torch.sum(aspect, dim=1)
        # aspect_pool = torch.div(aspect_pool, aspect_len.view(aspect_len.size(0), 1))
        #
        # text_raw_len = torch.tensor(context_len, dtype=torch.float).to(self.opt.device)
        # context_pool = torch.sum(context, dim=1)
        # context_pool = torch.div(context_pool, text_raw_len.view(text_raw_len.size(0), 1))
        #
        # aspect_final, _ = self.attention_aspect(aspect, context_pool)
        # print('aspect_final',aspect_final.size())
        # aspect_final = aspect_final.squeeze(dim=1)
        # print('aspect_final_sqqq', aspect_final.size())
        # context_final, _ = self.attention_context(context, aspect_pool)
        # print('context_final', context_final.size())
        # context_final = context_final.squeeze(dim=1)
        # print('context_final_sqqq', context_final.size())
        #
        # att_weight = torch.cat((aspect_final, context_final), dim=-1)
        # print('text_out', context.size())
        # print('aspect_double_idx', aspect_double_idx.size())
        # print('text_len', context_len.size())
        #print('aspect_len', aspect_len.size())
        w = self.position_weight(context, aspect_double_idx, context_len, aspect_len)
        #print("w", w.size())
        x = F.relu(self.gc1(w, adj))
        #print("x1", x.size())
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, context_len, aspect_len), adj))
        #print("x2", x.size())
        x = self.mask(x, aspect_double_idx)
        #print("x3", x.size())
        alpha_mat = torch.matmul(x, context.transpose(1, 2))
        #print('alpha_mat', alpha_mat.size())
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        #print('alpha', alpha.size())
        x = torch.matmul(alpha, context).squeeze(1)  # batch_size x 2*hidden_dim
       # print('x', x.size())
        output = self.fc(x)
        return output
        # GCN Layer
        # x = F.relu(self.gc1(att_weight, adj))
        # x = F.relu(self.gc2(x, adj))
        # x = self.mask(x, aspect_double_idx)
        # alpha_mat = torch.matmul(x, context.transpose(1, 2))
        # alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        # x = torch.matmul(alpha, context).squeeze(1)  # batch_size x 2*hidden_dim


        # Output Layer
        out = self.dense(x)
        return out
