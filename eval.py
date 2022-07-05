#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   eval.py
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
import os
from sklearn.metrics import roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if __name__ == '__main__':
    # https://github.com/lansinuote/Huggingface_Toturials/blob/main/7.%E4%B8%AD%E6%96%87%E5%88%86%E7%B1%BB.ipynb
    BATCH_SIZE = 16
    NUM_WORKERS = 8
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT") # tomh/toxigen_hatebert
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    dataset = GabHateCorpus(tokenizer, mode=4)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn, shuffle=False, num_workers=NUM_WORKERS)
    model = AutoModelForSequenceClassification.from_pretrained("tomh/toxigen_hatebert")
    model = model.cuda()

    model.eval()
    # ACC
    correct = 0
    total = 0
    # AUC
    label_all = []
    prob_all = []

    for i, (input_ids, attention_mask, labels, implicit_labels) in tqdm(enumerate(loader)):
        with torch.no_grad():
            out = model(input_ids=input_ids.to(device),
                        attention_mask=attention_mask.to(device))

        out = out.logits.argmax(dim=1)
        correct += (out.to('cpu') == labels).sum().item()
        total += len(labels)

        label_all.extend(labels)
        prob_all.extend(out.cpu())

    print("ACC", correct / total)
    print("AUC",roc_auc_score(label_all, prob_all))





