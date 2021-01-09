import re
import csv
import string
import pandas as pd

def remove_punct(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)

def find_nth(sent, aspect, n):
    count = 0
    words = sent.split(' ')
    for j in range(len(words)):
        if words[j] == aspect:
            count = count + 1
            index = sum(len(x) + 1 for i, x in enumerate(words)
                        if i < j)
            if (count == n):
                return index

def process_input(raw_input):
    content, aspect, start, end = list(), list(), list(), list()
    with open(raw_input, 'r', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        j = 0
        count=0
        for row in reader:
            if (j == 0):
                j = 1
            else:
                sent = row[0].lower()
                print(sent)
                sent = remove_punct(sent)
                sent.replace('\d+', '')
                sent = re.sub(r"^\s+|\s+$", "", sent)

                aspects = [x.replace("'", "").replace('[', "").replace("\"", "").replace(']', "").strip().lower()
                           for x
                           in row[1].split(",")]

                while ("" in aspects):
                    aspects.remove("")

                sentiments = [x.strip().replace("'", "").replace('[', "").replace("\"", "").replace(']', "").lower()
                              for
                              x in row[2].split(",")]
                while ("" in sentiments):
                    sentiments.remove("")

                for i in range(0, len(aspects)):
                    _aspect = aspects[i]
                    _aspect = remove_punct(_aspect)
                    _aspect.replace('\d+', '')
                    _aspect = re.sub(r"^\s+|\s+$", "", _aspect)
                    if len(_aspect.split()) > 1:
                        start_index = sent.find(_aspect)
                        if start_index == -1:
                            print(sent + " - " + _aspect)
                            continue
                        end_index = start_index + len(_aspect)
                    else:
                        if not (any(char.isdigit() for char in aspects[i])):
                            start_index = find_nth(sent, _aspect, 1)
                            if start_index is None:
                                count = count + 1
                                continue
                            end_index = start_index + len(_aspect)
                        else:
                            _aspect = aspects[i][:-2]
                            start_index = find_nth(sent, _aspect, int(aspects[i][-1]))
                            if start_index is None:
                                continue
                            end_index = start_index + len(aspects[i][:-2])

                    content.append(sent)
                    aspect.append(_aspect)
                    start.append(start_index)
                    end.append(end_index)
                    sentiment=0

    test_data = {'content': content, 'aspect': aspect, 'sentimet':sentiment, 'from': start, 'to': end}
    print(aspect)
    test_data = pd.DataFrame(test_data, columns=test_data.keys())
    test_data.to_csv(('./datasets/semeval14/processed.csv'), index=None)