# stacked generalization with linear meta model on blobs dataset
import csv
from numpy import dstack


import process_input
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
    prob = model.predict_proba(stackedX)
    return yhat,prob


def get_input(text,petitioner,defendant):
    isEmptyPet = False
    # text = input("Enter the sentence: ")
    # petitioner = str(input("Enter Petitioner Party Member/s: "))
    # defendant = str(input("Enter Defendant Party Member/s: "))
    text=text
    petitioner=petitioner
    defendant=defendant
    # print(text)
    # print(petitioner)
    # print(defendant)
    if (petitioner == ''):
        isEmptyPet = True

    petitioner_list = petitioner.split(",")
    defendant_list = defendant.split(",")
    pet_count = len(petitioner_list)

    # print(f"pet count {pet_count}")

    party = f"[{petitioner_list},{defendant_list}]"

    csv_file = './user_input/raw_input.csv'

    with open(csv_file, 'w', newline='') as input_file:
        writer = csv.writer(input_file)
        writer.writerow(['Sentence', 'party', 'Sentiment'])
        writer.writerow([text, party, 0])

    process_input.process_input(csv_file)

    return pet_count,isEmptyPet



def getoverall_sentiment(positive,negative,neutral):
    if (len(positive)!=0 or len(negative)!=0):
        if (len(positive)>len(negative)):
            result= 2
        elif(len(positive)<len(negative)):
            result=0
        else:
            count=len(positive)
            if(count==1):
                if(positive[0]>negative[0]):
                    result=2
                else:
                    result=0
            else:
                pos_avg= sum(positive)/count
                neg_avg=sum(negative)/count
                if (pos_avg> neg_avg):
                    result = 2
                else:
                    result = 0
    else:
        result=1
    return result


def pred(text,petitioner,defendant,models_list, opt_list, tokenizer,loaded_model):
    isEmptyPet = False
    # text = input("Enter the sentence: ")
    # petitioner = str(input("Enter Petitioner Party member/s: "))
    # defendant = str(input("Enter Defendant Party Member/s: "))

    if (petitioner == ''):
        isEmptyPet = True

    petitioner_list = petitioner.split(",")
    defendant_list = defendant.split(",")

    aspects = []
    if (len(petitioner_list) > 0 and petitioner_list[0] != ''):
        for i in petitioner_list:
            aspects.append(i)
    if (len(defendant_list) > 0 and defendant_list[0] != ''):
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

    pred_list = get_predictlist(models_list, opt_list, tokenizer)

    # evaluate model on test set
    yhat, prob = stacked_prediction(pred_list, loaded_model)
    # print ("prediction.......",yhat)
    # acc = accuracy_score(testy, yhat)
    # print('Stacked Test Accuracy: %.3f' % acc)
    # f1 = f1_score(testy, yhat, average='macro')
    # print('Stacked f1 score: %.3f' % f1)
    class_names = ['Negative', 'Neutral', 'Positive']
    print("-------------------------Results----------------------------------------------------")
    print(("Sentence : {}".format(text)))
    pet_dict = {}
    def_dict = {}
    pet_positive = []
    pet_negative = []
    pet_neutral = []
    def_positive = []
    def_negative = []
    def_neutral = []
    pet_flag = 0
    def_flag = 0
    for i in range(len(yhat)):
        if (not isEmptyPet):
            if (pet_flag == 0):
                print("Sentiments for Petitioner--->")
                pet_flag = 1
            if (i < pet_count):
                print(("    {} - {}".format(aspects[i], class_names[yhat[i]]))),
                pet_dict[aspects[i]] = class_names[yhat[i]]
                if (yhat[i] == 0):
                    pet_negative.append(prob[i][0])
                elif (yhat[i] == 1):
                    pet_neutral.append(prob[i][0])
                else:
                    pet_positive.append(prob[i][2])
            else:
                if (def_flag == 0):
                    print("Sentiments for Defendant--->")
                    def_flag = 1
                print(("   {} - {}".format(aspects[i], class_names[yhat[i]]))),
                def_dict[aspects[i]] = class_names[yhat[i]]
                if (yhat[i] == 0):
                    def_negative.append(prob[i][0])
                elif (yhat[i] == 1):
                    def_neutral.append(prob[i][0])
                else:
                    def_positive.append(prob[i][2])
        else:
            if (def_flag == 0):
                print("Sentiments for Defendant--->")
                def_flag = 1
            print(("   {} - {}".format(aspects[i], class_names[yhat[i]]))),
            def_dict[aspects[i]] = class_names[yhat[i]]
        pet_overall = getoverall_sentiment(pet_positive, pet_negative, pet_neutral)
        def_overall = getoverall_sentiment(def_positive, def_negative, def_neutral)
        print("Overall Sentiments for Petitioner--->")
        print(("    {} - {}".format("Petitioner", class_names[pet_overall]))),
        print("Overall Sentiments for Defendant--->")
        print(("    {} - {}".format("Defendant", class_names[def_overall]))),

    return text, pet_dict, def_dict, class_names[pet_overall], class_names[def_overall]

if __name__ == '__main__':
    models = ["bert_spc", 'bert_atae_lstm', "gcn_bert", "ram_bert", "lcf_bert"]
    # loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    # models_list,opt_list,tokenizer= get_model(models)

    # while (True):
    #     print('\n')
    #     print("-----------------------------------------------------------")
    #     print('\n')
    #     isEmptyPet = False
    #     text = input("Enter the sentence: ")
    #     petitioner = str(input("Enter Petitioner Party member/s: "))
    #     defendant = str(input("Enter Defendant Party Member/s: "))
    #
    #
    #     if (petitioner == ''):
    #         isEmptyPet = True
    #
    #     petitioner_list = petitioner.split(",")
    #     defendant_list = defendant.split(",")
    #
    #     aspects=[]
    #     if (len(petitioner_list)>0):
    #         for i in petitioner_list:
    #             aspects.append(i)
    #     if (len(defendant_list)>0):
    #         for j in defendant_list:
    #             aspects.append(j)
    #     pet_count = len(petitioner_list)
    #
    #     party = f"[{petitioner_list},{defendant_list}]"
    #
    #     csv_file = '/raw_input.csv'
    #
    #     with open(csv_file, 'w', newline='') as input_file:
    #         writer = csv.writer(input_file)
    #         writer.writerow(['Sentence', 'party', 'Sentiment'])
    #         writer.writerow([text, party, 0])
    #
    #     process_input.process_input(csv_file)
    #
    #     pred_list= get_predictlist(models_list,opt_list,tokenizer)
    #
    #     # evaluate model on test set
    #     yhat = stacked_prediction(pred_list, loaded_model)
    #     #print ("prediction.......",yhat)
    #     #acc = accuracy_score(testy, yhat)
    #     # print('Stacked Test Accuracy: %.3f' % acc)
    #     # f1 = f1_score(testy, yhat, average='macro')
    #     # print('Stacked f1 score: %.3f' % f1)
    #     class_names = ['Negative', 'Neutral', 'Positive']
    #     print("-------------------------Results----------------------------------------------------")
    #     print(("Sentence : {}".format(text)))
    #
    #     pet_flag = 0
    #     def_flag = 0
    #     for i in range(len(yhat)):
    #         if (not isEmptyPet):
    #             if (pet_flag == 0):
    #                 print("Sentiments for Petitioner--->")
    #                 pet_flag = 1
    #             if (i < pet_count):
    #                 print(("    {} - {}".format(aspects[i], class_names[yhat[i]]))),
    #             else:
    #                 if (def_flag == 0):
    #                     print("Sentiments for Defendant--->")
    #                     def_flag = 1
    #                 print(("   {} - {}".format(aspects[i], class_names[yhat[i]]))),
    #         else:
    #             if (def_flag == 0):
    #                 print("Sentiments for Defendant--->")
    #                 def_flag = 1
    #             print(("   {} - {}".format(aspects[i], class_names[yhat[i]]))),
