import pickle

from flask import Flask, request, url_for, redirect, render_template
import PredictForRawSentence as pr
from ensembleonepredict import get_predictlist
import ensembleOne
from ensembleonepredict import get_model
app = Flask(__name__)

global tokenizer
global opt
global model

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')


@app.route('/result',methods=["POST"])
def result():
    # if request.method == 'POST':
    text = request.form.get('string')  # access the data inside
    petitioner = request.form.get('pet_members')
    defendant = request.form.get('def_members')
    result_list =list()
    print(petitioner)
    print(defendant)

    sentence, pet_dict, def_dict, pet_overall, def_overall = ensembleOne.pred(text,petitioner,defendant,models_list, opt_list, tokenizer,loaded_model)

    result_list.append(text)
    result_list.append(pet_dict)
    result_list.append(def_dict)
    result_list.append(pet_overall)
    result_list.append(def_overall)

    result=result_list

    return render_template('result.html',result=result)


if __name__ == '__main__':
    models = ["bert_spc", 'bert_atae_lstm', "gcn_bert", "ram_bert", "lcf_bert"]
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    models_list, opt_list, tokenizer = get_model(models)

    app.run(debug=True)
