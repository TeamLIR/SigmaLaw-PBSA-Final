# stacked generalization with linear meta model on blobs dataset
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from numpy import dstack
import numpy as np
import os
from sklearn.metrics import f1_score
from predict import main


# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members):
    stackX = None
    for i in members:
        yhat= main(i)
        print(yhat.size())

        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = dstack((stackX, yhat))
    # flatten predictions to [rows, members x probabilities]
    print(type(stackX))
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))
    return stackX


# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputy):
    # create dataset using ensemble
    stackedX = stacked_dataset(members)
    # fit standalone model
    model = LogisticRegression()
    model.fit(stackedX, inputy)
    return model




# make a prediction with the stacked model
def stacked_prediction(members, model):
    # create dataset using ensemble
    stackedX = stacked_dataset(members)
    # make a prediction
    yhat = model.predict(stackedX)
    return yhat



if __name__ == '__main__':
    train_data = pd.read_csv('./datasets/semeval14/test.csv', header=0, index_col=None)
    models = [ 'bert_atae_lstm',"gcn_bert","ram_bert", "bert_spc","aen_bert","lcf_bert"]
    a=train_data['sentiment'].values.tolist()
    train_data['sentiment'] += 1
    b = train_data['sentiment'].values.tolist()
    testy = np.array(train_data['sentiment'].values.tolist())
    model = fit_stacked_model(models, testy)
    # evaluate model on test set
    yhat = stacked_prediction(models, model)
    acc = accuracy_score(testy, yhat)
    print('Stacked Test Accuracy: %.3f' % acc)
    f1 = f1_score(testy, yhat, average='macro')
    print('Stacked f1 score: %.3f' % f1)
