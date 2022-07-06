#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   eval_bert.py
@Contact :   1720613556@qq.com
@License :   (C)Copyright 2021-2022

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/7/4 16:12   dst      1.0         None
'''
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.GabHateCorpus import GabHateCorpus
from  datasets.ImplicitHateCorpus import ImplicitHateCorpus
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import os

os.environ["CUDA_VISIBLE_DEVICES"]='0'

def evaluate(labels_all, predict_all, eval_all=False):
    """evaluation"""
    acc = accuracy_score(labels_all, predict_all)
    print("Acc:", acc)
    if eval_all:
        target_names = ['toxic', 'non-toxic']
        print(classification_report(labels_all, predict_all, target_names=target_names))
        print("Auc:", roc_auc_score(labels_all, predict_all))

def load_dataset(DATASET, tokenizer, mode):
    if DATASET == 'GabHateCorpus':
        dataset = GabHateCorpus(tokenizer, mode=mode)
    elif DATASET == 'ImplicitHateCorpus':
        dataset = ImplicitHateCorpus(tokenizer, mode=mode)
    return dataset

def prepare_model(tokenizer_name, model_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)  # tomh/toxigen_hatebert
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

if __name__ == '__main__':
    # https://github.com/lansinuote/Huggingface_Toturials/blob/main/7.%E4%B8%AD%E6%96%87%E5%88%86%E7%B1%BB.ipynb

    BATCH_SIZE = 128
    NUM_WORKERS = 8
    # MODE:
    #         1. IM
    #         2. EX
    #         3. None
    #         4. EX + IM
    #         5. EX +IM + NON
    MODE = 5
    DATASET = 'GabHateCorpus'
    TOKENIZER = "./models/hateBERT"
    MODEL_NAME = "./models/hateBERT"
    POS_LABEL = "toxic"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using Device: {device}")

    # MODEL
    # 1. hateBert: GroNLP/hateBERT
    # 2. tomh/toxigen_roberta
    # 3. tomh/toxigen_hatebert
    # 4. cardiffnlp/twitter-roberta-base-hate
    tokenizer, model = prepare_model(TOKENIZER, MODEL_NAME)
    model = model.to(device)
    model.eval()


    # DATA
    dataset = load_dataset(DATASET, tokenizer, mode=MODE) # 调整mode
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn, shuffle=False, num_workers=NUM_WORKERS)


    label_all = []
    pred_all = []
    # EVAL
    for i, (input_ids, attention_mask, labels, implicit_labels) in tqdm(enumerate(loader), total=len(loader)):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask)

        # fixme: huggingface的这些模型toxic是正例还是负例是混乱的(我感觉2和4把non toxic作为正例了)
        if POS_LABEL=="toxic":
            out = out.logits.argmax(dim=1).cpu()
        else:
            out = out.logits.argmin(dim=1).cpu()
        label_all.extend(labels)
        pred_all.extend(out)
    evaluate(label_all, pred_all, eval_all=True) if MODE > 4 else evaluate(label_all, pred_all, eval_all=False)






