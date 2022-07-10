#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Perspective.py
@Contact :   1720613556@qq.com
@License :   (C)Copyright 2021-2022

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/6/27 11:31   dst      1.0         None
'''
import json
import os
import time

import pandas as pd
import requests
from perspective import PerspectiveAPI
from tqdm import tqdm
API_KEY = "AIzaSyA2dVIbryh2AOqGtNWvne-za_9pUV-Wy7o"
url = ("https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze" + "?key=" + API_KEY)

def get_toxicity(text):
    p = PerspectiveAPI(API_KEY)
    result = p.score(text)
    # print("Toxicity score: ", result["TOXICITY"])
    return result["TOXICITY"]


def score_generations(generations_filename, scores_filename):
    # csv columns
    categories = ["toxicity", "severe_toxicity", "insult", "flirtation", \
                  "profanity", "sexually_explicit", "threat", "identity_attack"]
    generations_columns = ["text"] + categories
    resume_mode = False
    if os.path.exists(scores_filename):
        generations_df = pd.read_csv(scores_filename)
        resume_mode = True
    else:
        generations_df = pd.DataFrame(columns=generations_columns)
        generations_df.to_csv(scores_filename, index=False)

    num_requests = 0
    with open(generations_filename, 'r', encoding="utf-8") as f:
        all_texts = f.readlines()
        # resume
        if resume_mode:
            # 保存的最后一个
            last_text = generations_df.text.values[-1]
            # try to find last text
            start_idx = all_texts.index(last_text + '\n') + 1
            assert  start_idx < len(all_texts)
            all_texts = all_texts[start_idx+1:]
            print("resume mode activated. starting from:", start_idx)
            generations_df = pd.DataFrame(columns=generations_columns) # 内存中df清空

        for line in tqdm(all_texts, total=len(all_texts)):
            if num_requests % 10 == 0:
               time.sleep(0.2)
            if num_requests == 50000:
                break

            if num_requests % 500 == 0:
                print("{} | Request #{}".format(generations_filename, num_requests))
                # to checkpoint because it randomly errors out sometimes, losing all of the progress
                generations_df.to_csv(scores_filename, mode='a', header=False, index=False)
                generations_df = pd.DataFrame(columns=generations_columns) # 内存中df清空

            if line[-1] == "\n":
                line = line[:-1]

            data_dict = {"comment": {"text": line},
                         "languages": ["en"],
                         "requestedAttributes": {"TOXICITY": {}, "SEVERE_TOXICITY": {}, "INSULT": {}, \
                                                 "FLIRTATION": {}, "PROFANITY": {}, "SEXUALLY_EXPLICIT": {}, \
                                                 "THREAT": {}, "IDENTITY_ATTACK": {}}}

            if len(line) == 0:
                continue
            else:
                response = requests.post(url=url, data=json.dumps(data_dict))
                response_dict = json.loads(response.content)
                if "attributeScores" in response_dict:
                    row={}
                    for categ in categories:
                        score = response_dict["attributeScores"][categ.upper()]["spanScores"][0]["score"]["value"]
                        row[categ] = score
                    row['text'] = line
                    generations_df = generations_df.append(row, ignore_index=True)
                    num_requests += 1
                else:
                    print(response_dict)

    generations_df.to_csv(scores_filename, mode='a', header=False, index=False)


if __name__ == '__main__':
    #score_generations("./data/GabHate/implicit.txt", r"C:\Users\123\Desktop\GabHateCorpus\perspective\implicit.csv")
    # score_generations("./data/GabHate/explicit.txt", r"C:\Users\123\Desktop\GabHateCorpus\perspective\explicit.csv")
    # score_generations("./data/GabHate/non_toxic.txt", r"C:\Users\123\Desktop\GabHateCorpus\perspective\non_toxic.csv")
    score_generations("./data/ImplicitHate/implicit.txt", r"C:\Users\123\Desktop\implicit-hate-corpus\perspective\implicit.csv")
    score_generations("./data/ImplicitHate/explicit.txt", r"C:\Users\123\Desktop\implicit-hate-corpus\perspective\explicit.csv")
    score_generations("./data/ImplicitHate/non_toxic.txt", r"C:\Users\123\Desktop\implicit-hate-corpus\perspective\non_toxic.csv")