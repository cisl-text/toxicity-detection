import pandas as pd
from utils import evaluate
import os

def read_data(folder_dir, mode):
    if mode=="toxic":
        path = os.path.join(folder_dir,"toxic.csv")
    else:
        path = os.path.join(folder_dir,"non_toxic.csv")

    predict_all=[]
    labels_all=[]
    prob_all=[]
    data = pd.read_csv(path)
    for item in data.values:
        prob_all.append(item[1])
        predict_all.append(0 if item[1]<0.5 else 1)
        labels_all.append(1 if mode=='toxic' else 0)
    return labels_all, predict_all, prob_all

if __name__ == '__main__':
    predict_all=[]
    labels_all=[]
    prob_all=[]

    labels, predict, prob = read_data("perspective/toxigen","toxic")
    labels_all.extend(labels)
    predict_all.extend(predict)
    prob_all.extend(prob)
    labels, predict, prob = read_data("perspective/toxigen","none")
    labels_all.extend(labels)
    predict_all.extend(predict)
    prob_all.extend(prob)

    evaluate(labels_all, predict_all, POS_LABEL="toxic", eval_all=True, prob_all=prob_all)