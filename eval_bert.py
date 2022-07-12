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
import argparse
from sklearn.tree import export_text
import pandas as pd

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.GabHateCorpus import GabHateCorpus
from datasets.ImplicitHateCorpus import ImplicitHateCorpus
from datasets.ToxigenCorpus import ToxigenCorpus
import os
from utils import evaluate, get_config


def load_dataset(DATASET, tokenizer, mode, export = False):
    if DATASET == 'GabHateCorpus':
        dataset = GabHateCorpus(tokenizer, mode=mode, export=export)
    elif DATASET == 'ImplicitHateCorpus':
        dataset = ImplicitHateCorpus(tokenizer, mode=mode, export=export)
    elif DATASET == 'ToxigenCorpus':
        dataset = ToxigenCorpus(tokenizer, mode=mode, export=export)
    return dataset


def prepare_model(tokenizer_name, model_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)  # tomh/toxigen_hatebert
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def save_errors(save_data, save_path="./error_data/test.csv"):
    save_data = pd.DataFrame(save_data, columns=['text', 'predict_score', 'implicit'])
    save_data.to_csv(save_path, index=False)


if __name__ == '__main__':
    # https://github.com/lansinuote/Huggingface_Toturials/blob/main/7.%E4%B8%AD%E6%96%87%E5%88%86%E7%B1%BB.ipynb
    parser = argparse.ArgumentParser(description='bert classification')
    parser.add_argument("-c", "--config", type=str, default="./configs/eval_bert_config.yaml")
    args = parser.parse_args()
    config = get_config(args.config, mode='eval')
    os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda']

    BATCH_SIZE = config['batch_size']
    NUM_WORKERS = config['num_workers']
    # MODE:
    #         0. VO
    #         1. IM
    #         2. EX
    #         3. None
    #         4. EX + IM
    #         5. EX +IM + NON
    MODE = config['dataset']['mode']
    DATASET = config['dataset']['name']
    TOKENIZER = config['tokenizer']
    MODEL_NAME = config['model']
    POS_LABEL = config['dataset']['pos_label']
    EXPORT = config['export_negtive']
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
    dataset = load_dataset(DATASET, tokenizer, mode=MODE, export = EXPORT)  # 调整mode
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn, shuffle=False,
                        num_workers=NUM_WORKERS)
    print(f"the size of {DATASET}: {len(dataset)}")

    label_all = []
    pred_all = []
    pred_prob_all = []
    # EVAL
    if EXPORT == True: 
        export_text=[]
        for i, (input_ids, attention_mask, labels, implicit_labels, text) in tqdm(enumerate(loader), total=len(loader)):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            out = out.logits.cpu()
            pred_prob_all.extend(out[:, 1])
            if POS_LABEL == "toxic":
                out = out.argmax(dim=1)
            else:
                out = out.argmin(dim=1)
            for j in range(len(out)):
                if out[j]!=labels[j]:
                    export_text.append([text[j],pred_prob_all[j].item(),implicit_labels[j].item()])
        save_errors(export_text,f"./error_data/{DATASET}{MODE}_{MODEL_NAME}.csv")
    else: 
        for i, (input_ids, attention_mask, labels, implicit_labels) in tqdm(enumerate(loader), total=len(loader)):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids,
                            attention_mask=attention_mask)

            out = out.logits.cpu()
            pred_prob_all.extend(out[:, 1])
            if POS_LABEL == "toxic":
                out = out.argmax(dim=1)
            else:
                out = out.argmin(dim=1)
            label_all.extend(labels)
            pred_all.extend(out)
        evaluate(label_all, pred_all, POS_LABEL, eval_all=True, prob_all=pred_prob_all) if MODE > 4 else evaluate(label_all,
                                                                                                              pred_all,
                                                                                                              POS_LABEL,
                                                                                                              eval_all=False)
