import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class Bilinear_Attention(nn.Module):
    def __init__(self, n_hidden):
        """
        :param n_hidden:
        :param l2_reg:
        :param random_base:
        :param layer_id:
        """
        super(Bilinear_Attention, self).__init__()
        self.n_hidden = n_hidden
        # self.l2_reg = l2_reg
        # self.random_base = random_base
        # self.layer_id = layer_id
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))

    def forward(self, inputs, attend, length):
        """
        :param inputs: batch * max_len * n_hidden
        :param attend: batch * n_hidden
        :param length:
        :return:
        """
        batch_size = inputs.size()[0]
        max_len = inputs.size()[1]
        l= [max_len]

        # print("batch size: ", batch_size)
        # print("max_len: ", max_len)
        # w = tf.get_variable(
        #     name='att_w_' + str(layer_id),
        #     shape=[n_hidden, n_hidden],
        #     initializer=tf.random_uniform_initializer(-random_base, random_base),
        #     regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
        # )
        inputs = torch.reshape(inputs, [-1, self.n_hidden])
        # print("inputs: ", inputs.size())
        tmp = torch.reshape(torch.matmul(inputs, self.weight), [-1, max_len, self.n_hidden])
        # print("tmp: ", tmp.size())
        attend = torch.unsqueeze(attend, 2)
        # print("attend: ", attend.size())
        tmp = torch.reshape(torch.matmul(tmp, attend), [batch_size, 1, max_len])

        return F.softmax(tmp.type(torch.float32), dim=1)

        # #sofmax with len
        # inputs = tmp.type(torch.float32)
        # max_axis = torch.max(inputs, -1, keepdim=True, out=None)
        # inputs = torch.exp(inputs)
        # length = torch.reshape(length, [-1])
        # mask_k = torch.masked_select(length, torch.BoolTensor(l))
        # mask = torch.reshape(mask_k.type(torch.float32), inputs.size())
        # inputs *= mask
        # _sum = torch.sum(inputs, dim=-1, keep_dims=True) + 1e-9
        # return inputs / _sum


    #     return self.softmax_with_len(tmp, length, max_len)
    #
    # def softmax_with_len(inputs, length, max_len):
    #     inputs = inputs.type(torch.float32)
    #     max_axis = torch.max(inputs, -1, keep_dims=True)
    #     inputs = torch.exp(inputs - max_axis)
    #     length = torch.reshape(length, [-1])
    #     mask_k = torch.masked_select(length, max_len)
    #     mask = torch.reshape(mask_k.type(torch.float32), torch.size(inputs))
    #     inputs *= mask
    #     _sum = torch.sum(inputs, dim=-1, keep_dims=True) + 1e-9
    #     return inputs / _sum



if __name__ == '__main__':
    a = torch.randn(1, 3)
    print(a)
    b= torch.max(a)
    print(b)
