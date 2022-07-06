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
from  datasets.ImplicitHateCorpus import ImplicitHateCorpus
import os

os.environ["CUDA_VISIBLE_DEVICES"]='2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using Device: {device}")
if __name__ == '__main__':
    # https://github.com/lansinuote/Huggingface_Toturials/blob/main/7.%E4%B8%AD%E6%96%87%E5%88%86%E7%B1%BB.ipynb

    BATCH_SIZE = 16
    NUM_WORKERS = 4
    DATASET = 'ImplicitHateCorpus'
    tokenizer = AutoTokenizer.from_pretrained("tomh/toxigen_roberta") # tomh/toxigen_hatebert
    if DATASET == 'GabHateCorpus':
        dataset = GabHateCorpus(tokenizer, mode=3)
    elif DATASET == 'ImplicitHateCorpus':
        dataset = ImplicitHateCorpus(tokenizer, mode=1)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn, shuffle=False, num_workers=NUM_WORKERS)
    model = AutoModelForSequenceClassification.from_pretrained("tomh/toxigen_roberta").to(device)

    model.eval()
    correct = 0
    total = 0
    for i, (input_ids, attention_mask, labels, implicit_labels) in tqdm(enumerate(loader), total=len(loader)):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask)
        print(out.size)
        out = out.logits.argmax(dim=1).cpu()
        correct += (out == labels).sum().item()
        total += len(labels)

    print("ACC", correct / total)





