# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_transformers import BertTokenizer
import string
import re
import csv
import seaborn as sns


def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec

def _load_word_vec(data_path, vocab=None):
    embedding_model = {}
    f = open(data_path, 'r', encoding="utf8")
    for line in f:
        values = line.split()
        word = ''.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embedding_model[word] = coefs
    f.close()
    return embedding_model


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.840B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len
        # sample_test="I am Dilini. Who are you?"
        # tokens=self.tokenizer.tokenize(sample_test)
        # print('tokens are: {}'.format(len(tokens)))
        # print('token ids are: {}'.format(len(self.tokenizer.convert_tokens_to_ids(tokens))))

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


def remove_punct(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)



class ABSADataset(Dataset):
    def __init__(self, fname,yname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        fin = open(yname, 'rb')
        idx2gragh = pickle.load(fin)
        fin.close()

        all_data = []
        sen_len=[]
        target_len=[]
        with open(fname, 'r', encoding="utf8") as csvfile:
            aspectreader = csv.reader(csvfile, delimiter=',')
            j = 0
            count = 0
            input=[]
            examples = []
            position=[]
            m=0
            for row in aspectreader:
                if j == 0:
                    j = 1
                else:
                    sent = row[0].lower()
                    # print(sent)

                    sent = remove_punct(sent)
                    sent.replace('\d+', '')
                    # sent.replace(r'\b\w\b', '').replace(r'\s+', ' ')
                    # sent.replace('\s+', ' ', regex=True)
                    # sent=re.sub(r"^\s+|\s+$", "", sent), sep='')
                    sent = re.sub(r"^\s+|\s+$", "", sent)
                    input.append(sent)
                    examples.append(sent)
                    sen_words = len(sent.split(" "))

                    # nb_aspects = int(row[1])
                    aspect = row[1].lower()
                    examples.append(aspect)
                    aspect_words = len(aspect.split(" "))

                    start = row[3]
                    end = row[4]
                    polarity = row[2]
                    examples.append(polarity)
                    # print("****")
                    dependency_graph = idx2gragh[m]
                    text_left = sent[0:int(start) - 1]
                    text_right = sent[int(end) + 1:]

                    m += 3
                    text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
                    text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
                    text_left_indices = tokenizer.text_to_sequence(text_left)
                    text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
                    text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
                    text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right,
                                                                                reverse=True)
                    aspect_indices = tokenizer.text_to_sequence(aspect)
                    left_context_len = np.sum(text_left_indices != 0)
                    aspect_len = np.sum(aspect_indices != 0)
                    aspect_in_text = torch.tensor([left_context_len.item(), (left_context_len + aspect_len - 1).item()])
                    polarity = int(polarity) + 1

                    text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
                    bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
                    bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)

                    text_raw_bert_indices = tokenizer.text_to_sequence(
                        "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
                    aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

                    sent_new = sent[0:int(start)] + '$T$' + sent[int(end):]
                    i_input = sent_new.strip().split(' ')
                    for index_j, j in enumerate(i_input):
                        if "$T$" in j:
                            i_input[index_j] = '$T$'
                    i_target = aspect.split(' ')
                    len_input = len(i_input)
                    len_target = len(i_target)
                    target_position = i_input.index("$T$")
                    target_b_len = target_position
                    target_m_len = len_target
                    target_e_len = len_input - target_position - 1
                    target_b_list = list(range(1, target_b_len + 1))
                    target_b_list.reverse()
                    target_m_list = [0 for j in range(target_m_len)]
                    target_e_list = list(range(1, target_e_len + 1))

                    # 让距离太远的变正0
                    Ls = len(target_b_list + target_m_list + target_e_list)
                    for index_j, j in enumerate(target_b_list):
                        if j > 10:
                            target_b_list[index_j] = Ls
                    for index_j, j in enumerate(target_e_list):
                        if j > 10:
                            target_e_list[index_j] = Ls

                    i_position = target_b_list + target_m_list + target_e_list
                    i_position_encoder = [(1 - j / Ls) for j in i_position]
                    i_position_encoder = i_position_encoder + [0] * (tokenizer.max_seq_len - len(i_position))
                    position.append(i_position_encoder)
                    size= dependency_graph.shape
                    dependency_graph=np.pad(dependency_graph, ((0, tokenizer.max_seq_len - dependency_graph.shape[0]), (0, tokenizer.max_seq_len - dependency_graph.shape[0])), 'constant')
                    data = {
                        'raw_sentence': sent,
                        'aspect': aspect,
                        'text_bert_indices': text_bert_indices,
                        'bert_segments_ids': bert_segments_ids,
                        'text_raw_bert_indices': text_raw_bert_indices,
                        'aspect_bert_indices': aspect_bert_indices,
                        'text_raw_indices': text_raw_indices,
                        'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                        'text_left_indices': text_left_indices,
                        'text_left_with_aspect_indices': text_left_with_aspect_indices,
                        'text_right_indices': text_right_indices,
                        'text_right_with_aspect_indices': text_right_with_aspect_indices,
                        'aspect_indices': aspect_indices,
                        'aspect_in_text': aspect_in_text,
                        'polarity': polarity,
                        'position' : i_position_encoder,
                        'x_len' : sen_words,
                        'target_len': aspect_words,
                        'dependency_graph': dependency_graph,


                    }

                    all_data.append(data)
            position = np.array(position)
            all_sentence = [s for s in input]
            targets_nums = [all_sentence.count(s) for s in all_sentence]
            targets = []
            targets_len=[]
            target_len_all=[]
            i = 0
            while i < len(all_sentence):
                num = targets_nums[i]
                target = []
                target_length=[]
                len_list=[]
                for j in range(num):
                    e= examples[(i + j) * 3 + 1]
                    target_length.append(len(e.split(" ")))
                    f=tokenizer.text_to_sequence("[CLS] " + examples[(i + j) * 3 + 1]+ " [SEP]")
                    target.append(f)
                for k in range(8 - len(target_length)):
                    target_length.append(0)
                for v in range(8):
                    l=[]
                    if (target_length[v]!=0):
                        for k in range(v):
                            l.append(0)
                        l.append(target_length[v])
                        for m in range(8-v-1):
                            l.append(0)
                        len_list.append(np.array(l))
                    else:
                        len_list.append(np.zeros(8).astype(int))
                #target_len_all.append(len_list)



                for j in range(num):
                    targets.append(target)
                    targets_len.append(len_list)

                i = i + num
            targets_nums = np.array(targets_nums)


            #train_target_whichone = self.get__whichtarget(targets_nums, max_target_num)
           # targets_position = self.get_position_2(position, targets_nums, max_target_num)
            train_target_whichone = self.get__whichtarget(targets_nums, 8)
            targets_position = self.get_position_2(position, targets_nums, 8)
            for i in range(len(all_data)):
                all_data[i]["all_targets"] = self.get_targets_all(targets[i],tokenizer)
                all_data[i]["all_positions"] = np.array(targets_position[i])
                #all_data[i]['targets_num_max'] = 8
                all_data[i]['which_one']= np.array(train_target_whichone[i]).astype(int)
                all_data[i]['targets_len']=np.array(targets_len[i])

        self.data = all_data


    def get_targets_all(self,targets,tokenizer):
        sen_ids = []
        sen_lens = []
        for x in targets:
            sen_ids.append(x)
        for j in range(8 - len(targets)):
            n=np.asarray([0] * tokenizer.max_seq_len)
            sen_ids.append(np.asarray([0] * tokenizer.max_seq_len))
                #sen_len.append(0)
            #sen_ids.append(sen_id)
            #sen_lens.append(sen_len)


        return np.asarray(sen_ids)


    def get__whichtarget(self,targets_num, max_target_num ):
            '''
            :param target_num: a one dimension array:[1,2,2,1,...]
            :param max_target_num: max_target_num is 13 in Res data
            :return: which_one,shape = [?,max_target_num]:[[1,0,0,0,...],
                                                             [1,0,0,0,...],
                                                             [0,1,0,0,...],
                                                             [1,0,0,0,...],
                                                             ...]
            '''
            which_one = np.zeros((targets_num.shape[0], max_target_num))
            # 补上位置信息，如果是3，那就补上[1,0,0][0,1,0][0,0,1]
            # 做法：根据每个的数字，循环得到对于位置,当然序号加上该值
            i = 0
            while i < targets_num.shape[0]:
                for j in range(targets_num[i]):
                    which_one[i, j] = 1
                    i += 1
            return which_one.tolist()

    def get_position_2(self,target_position, targets_num, max_target_num):
        """
        结合输入的target_position以及target_num,target_num是多少，就由多少个，并且重复多少次。
        不足max_target_num的，补0.
        """
        positions = []
        i = 0
        while i < targets_num.shape[0]:
            i_position = []
            for t_num in range(targets_num[i]):
                i_position.append(target_position[i + t_num])

            for j in range(max_target_num - targets_num[i]):
                i_position.append(np.zeros([target_position.shape[1]]))
            for t_num in range(targets_num[i]):
                positions.append(i_position)
                i += 1

        return positions



    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    tokenizer = Tokenizer4Bert(80, 'bert-base-uncased')
    # dataset = ABSADataset('./datasets/semeval14/Restaurants_Train.xml.seg', tokenizer)
    dataset = ABSADataset('./datasets/semeval14/train.csv', tokenizer)
