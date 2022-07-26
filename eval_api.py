import pandas as pd
from utils import evaluate
import os

def read_data(folder_dir, mode):
    if mode=="toxic":
        path = os.path.join(folder_dir,"toxic.csv")
    elif mode == "implicit":
        path = os.path.join(folder_dir,"implicit.csv")
    elif mode == "explicit":
        path = os.path.join(folder_dir,"explicit.csv")
    else:
        path = os.path.join(folder_dir,"non_toxic.csv")

    predict_all=[]
    labels_all=[]
    prob_all=[]
    data = pd.read_csv(path)
    for item in data.toxicity.values:
        try:
            float(item)
        except:
            continue
        prob_all.append(float(item))
        predict_all.append(0 if float(item)<0.5 else 1)
        labels_all.append(0 if mode=='none' else 1)
    return labels_all, predict_all, prob_all

if __name__ == '__main__':
    predict_all=[]
    labels_all=[]
    prob_all=[]
    # implicit gab toxigen
    labels, predict, prob = read_data("perspective/gab","implicit")
    labels_all.extend(labels)
    predict_all.extend(predict)
    prob_all.extend(prob)
    labels, predict, prob = read_data("perspective/gab","explicit")
    labels_all.extend(labels)
    predict_all.extend(predict)
    prob_all.extend(prob)
    labels, predict, prob = read_data("perspective/gab","none")
    labels_all.extend(labels)
    predict_all.extend(predict)
    prob_all.extend(prob)

    evaluate(labels_all, predict_all, eval_all=True, prob_all=prob_all)