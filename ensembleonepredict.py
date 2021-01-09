import csv
import argparse
import random
import numpy
from pytorch_transformers import BertModel
import torch
from torch.utils.data import DataLoader, random_split
from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from models import LSTM, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, LCF_BERT
from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC
from models.sdgcn import SDGCN



class Preloader:
    def __init__(self, opt,tokenizer, bert):
        # self.tokernizer
        self.opt = opt
        self.model = opt.model_class(bert, opt).to(opt.device)
        model_path = 'saved/' + opt.model_name + '.hdf5'
        self.model.load_state_dict(torch.load(model_path))


    def get_model(self):
        return self.model


class Predictor:
    def __init__(self,opt,model,testset):
        self.opt = opt
        self.model=model
        self.testset = testset

    def get_prediction(self, data_loader):
        self.model.eval()

        review_texts = []
        prediction_probs = []
        real_values = []
        aspects = []

        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_sentence = t_sample_batched['raw_sentence']
                t_aspect = t_sample_batched['aspect']
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)
                # print(type(t_outputs))
                # print(t_outputs.shape)
                # print(t_outputs)

                review_texts.extend(t_sentence)
                # prediction_probs.extend(torch.argmax(t_outputs, -1))
                prediction_probs.extend(t_outputs)
                real_values.extend(t_targets)
                aspects.extend(t_aspect)

        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()


        return review_texts, prediction_probs, real_values, aspects

    def save_predictions(self):

        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)

        sentence, y_pred_probs, y_test, aspect = self.get_prediction(test_data_loader)
        class_names = [0, 1, 2]

        # csv_file = 'predictions/{}_predictions.csv'.format(self.opt.model_name)
        #
        # with open(csv_file, 'w', newline='') as pred_file:
        #     writer = csv.writer(pred_file)
        #     writer.writerow(['sentence', 'aspect', 'predicted sentiment', 'true sentiment'])
        #     for i in range(len(sentence)):
        #         writer.writerow([sentence[i], aspect[i], class_names[y_pred_probs[i]], class_names[y_test[i]]])


        return y_pred_probs


def main(model):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=model, type=str)
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
    parser.add_argument('--device', default='cpu', type=str, help='e.g. cuda:0')
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
        'gcn_bert': SDGCN,
        'bert_atae_lstm': ATAE_LSTM,
        'memnet': MemNet,
        'ram_bert': RAM,
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
        'bert_atae_lstm': ['text_raw_indices', 'aspect_indices'],
        'ian': ['text_raw_indices', 'aspect_indices'],
        'gcn_bert': ['text_raw_bert_indices', 'aspect_bert_indices'],
        'memnet': ['text_raw_without_aspect_indices', 'aspect_indices'],
        'ram_bert': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
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
    return (opt)


def get_model(models):
    opt_list = []
    models_list=[]
    for model in models:
        opt = main(model)
        opt_list.append(opt)

    tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
    bert = BertModel.from_pretrained(opt.pretrained_bert_name, output_hidden_states=True)

    for opt in opt_list:
        if (opt.model_name=="bert_spc" or opt.model_name=="lcf_bert"):
            bert1  = BertModel.from_pretrained(opt.pretrained_bert_name)
            pred= Preloader(opt, tokenizer, bert1)
            models_list.append(pred.get_model())

        else:
            pred = Preloader(opt, tokenizer, bert)
            models_list.append(pred.get_model())

    return models_list,opt_list,tokenizer

def get_predictlist(models,opt_list,tokenizer):
    pred_list = []
    testset = ABSADataset( './datasets/semeval14/processed.csv', tokenizer)
    for i in range(len(models)):
        pred= Predictor(opt_list[i],models[i],testset)
        predictions = pred.save_predictions()
        pred_list.append(predictions)
    return pred_list
