import csv
import argparse
import random
import numpy
from pytorch_transformers import BertModel
import torch
from torch.utils.data import DataLoader, random_split
from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from models import LSTM,  MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, LCF_BERT
from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC
from train import logger
import process_input
import re
import string
import pandas as pd

class Predictor:
    def __init__(self,opt):
        self.opt = opt
        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test1']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        self.testset = ABSADataset(opt.dataset_file['test1'], tokenizer)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))

        model_path = 'state_dict/bert_spc_law_val_acc0.5314.hdf5'# provide best model path
        self.model.load_state_dict(torch.load(model_path))

    def get_prediction(self,data_loader):
        self.model.eval()
        text = []
        prediction_probs = []
        aspects = []

        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_sentence = t_sample_batched['raw_sentence']
                t_aspect = t_sample_batched['aspect']
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_outputs = self.model(t_inputs)

                text.extend(t_sentence)
                prediction_probs.extend(torch.argmax(t_outputs, -1))
                aspects.extend(t_aspect)

        prediction_probs = torch.stack(prediction_probs).cpu()
        return text, prediction_probs, aspects

    def save_predictions(self,pet_count,isEmptyPet):

        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)

        sentence, y_pred_probs, aspect = self.get_prediction(test_data_loader)
        class_names = ['Negative', 'Neutral', 'Positive']

        print(("Sentence : {}".format(sentence[0])))

        pet_flag = 0
        def_flag = 0
        for i in range(len(sentence)):
            if(not isEmptyPet):
                if (pet_flag==0):
                    print("Sentiments for Petitioner--->")
                    pet_flag=1
                if (i < pet_count):
                    print(("    {} - {}".format(aspect[i],class_names[y_pred_probs[i]]))),
                else:
                    if(def_flag==0):
                        print("Sentiments for Defendant--->")
                        def_flag=1
                    print(("   {} - {}".format(aspect[i],class_names[y_pred_probs[i]]))),
            else:
                if (def_flag == 0):
                    print("Sentiments for Defendant--->")
                    def_flag = 1
                print(("   {} - {}".format(aspect[i], class_names[y_pred_probs[i]]))),

def main():
    isEmptyPet=False
    text = input("Enter the sentence: ")
    petitioner = str(input("Enter Petitioner Party member/s: "))
    defendant = str(input("Enter Defendant Party Member/s: "))

    if (petitioner == ''):
        isEmptyPet = True

    petitioner_list = petitioner.split(",")
    defendant_list = defendant.split(",")

    pet_count = len(petitioner_list)

    party = f"[{petitioner_list},{defendant_list}]"

    csv_file = './user_input/raw_input.csv'

    with open(csv_file, 'w', newline='') as input_file:
        writer = csv.writer(input_file)
        writer.writerow(['Sentence','party','Sentiment'])
        writer.writerow([text, party,0])

    process_input.process_input(csv_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_spc', type=str)
    parser.add_argument('--dataset', default='law', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float,
                        help='set ratio between 0 and 1 for validation support')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'tc_lstm': TC_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
        'lcf_bert': LCF_BERT,
        # default hyper-parameters for LCF-BERT model is as follws:
        # lr: 2e-5
        # l2: 1e-5
        # batch size: 16
        # num epochs: 5
    }
    dataset_files = {
        'law': {
            'train': './datasets/semeval14/train.csv',
            'test1': './datasets/semeval14/processed.csv'
        }
    }
    input_colses = {
        'lstm': ['text_raw_indices'],
        'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'tc_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices', 'aspect_indices'],
        'atae_lstm': ['text_raw_indices', 'aspect_indices'],
        'ian': ['text_raw_indices', 'aspect_indices'],
        'memnet': ['text_raw_without_aspect_indices', 'aspect_indices'],
        'ram': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'cabasc': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices',
                   'text_right_with_aspect_indices'],
        'tnet_lf': ['text_raw_indices', 'aspect_indices', 'aspect_in_text'],
        'aoa': ['text_raw_indices', 'aspect_indices'],
        'mgan': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'bert_spc': ['text_bert_indices', 'bert_segments_ids'],
        'aen_bert': ['text_raw_bert_indices', 'aspect_bert_indices'],
        'lcf_bert': ['text_bert_indices', 'bert_segments_ids', 'text_raw_bert_indices', 'aspect_bert_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    pred = Predictor(opt)
    pred.save_predictions(pet_count,isEmptyPet)

if __name__ == '__main__':
    main()
