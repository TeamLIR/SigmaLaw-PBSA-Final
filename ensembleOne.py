# stacked generalization with linear meta model on blobs dataset
import csv
import pickle
from numpy import dstack


import process_input
from ensembleonepredict import get_model
from ensembleonepredict import get_predictlist



# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(yhat):
    stackX = None

    for i in yhat:
        ypred=i

        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = i
        else:
            stackX = dstack((stackX, i))
    # flatten predictions to [rows, members x probabilities]

    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))
    return stackX


# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members):
    # create dataset using ensemble
    stackedX = stacked_dataset(members)
    # fit standalone model

    return stackedX




# make a prediction with the stacked model
def stacked_prediction(yhat, model):
    # create dataset using ensemble
    stackedX = stacked_dataset(yhat)
    # make a prediction
    yhat = model.predict(stackedX)
    return yhat



if __name__ == '__main__':
    models = ["bert_spc", 'bert_atae_lstm', "gcn_bert", "ram_bert", "lcf_bert"]
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    models_list,opt_list,tokenizer= get_model(models)

    while (True):
        print('\n')
        print("-----------------------------------------------------------")
        print('\n')
        isEmptyPet = False
        text = input("Enter the sentence: ")
        petitioner = str(input("Enter Petitioner Party member/s: "))
        defendant = str(input("Enter Defendant Party Member/s: "))


        if (petitioner == ''):
            isEmptyPet = True

        petitioner_list = petitioner.split(",")
        defendant_list = defendant.split(",")

        aspects=[]
        if (len(petitioner_list)>0):
            for i in petitioner_list:
                aspects.append(i)
        if (len(defendant_list)>0):
            for j in defendant_list:
                aspects.append(j)
        pet_count = len(petitioner_list)

        party = f"[{petitioner_list},{defendant_list}]"

        csv_file = '/raw_input.csv'

        with open(csv_file, 'w', newline='') as input_file:
            writer = csv.writer(input_file)
            writer.writerow(['Sentence', 'party', 'Sentiment'])
            writer.writerow([text, party, 0])

        process_input.process_input(csv_file)

        pred_list= get_predictlist(models_list,opt_list,tokenizer)

        # evaluate model on test set
        yhat = stacked_prediction(pred_list, loaded_model)
        #print ("prediction.......",yhat)
        #acc = accuracy_score(testy, yhat)
        # print('Stacked Test Accuracy: %.3f' % acc)
        # f1 = f1_score(testy, yhat, average='macro')
        # print('Stacked f1 score: %.3f' % f1)
        class_names = ['Negative', 'Neutral', 'Positive']
        print("-------------------------Results----------------------------------------------------")
        print(("Sentence : {}".format(text)))

        pet_flag = 0
        def_flag = 0
        for i in range(len(yhat)):
            if (not isEmptyPet):
                if (pet_flag == 0):
                    print("Sentiments for Petitioner--->")
                    pet_flag = 1
                if (i < pet_count):
                    print(("    {} - {}".format(aspects[i], class_names[yhat[i]]))),
                else:
                    if (def_flag == 0):
                        print("Sentiments for Defendant--->")
                        def_flag = 1
                    print(("   {} - {}".format(aspects[i], class_names[yhat[i]]))),
            else:
                if (def_flag == 0):
                    print("Sentiments for Defendant--->")
                    def_flag = 1
                print(("   {} - {}".format(aspects[i], class_names[yhat[i]]))),
