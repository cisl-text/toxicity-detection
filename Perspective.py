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

import pandas as pd
import requests
from perspective import PerspectiveAPI

API_KEY = "AIzaSyA2dVIbryh2AOqGtNWvne-za_9pUV-Wy7o"
url = ("https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze" + "?key=" + API_KEY)

def get_toxicity(text):
    p = PerspectiveAPI(API_KEY)
    result = p.score(text)
    # print("Toxicity score: ", result["TOXICITY"])
    return result["TOXICITY"]


def score_generations(generations_filename, scores_filename):
    categories = ["toxicity", "severe_toxicity", "insult", "flirtation", \
                  "profanity", "sexually_explicit", "threat", "identity_attack"]
    generations_columns = ["text"] + categories

    if os.path.exists(scores_filename):
        generations_df = pd.read_json(scores_filename, lines=True)
        start_row = generations_df.shape[0]
    else:
        generations_df = pd.DataFrame(columns=generations_columns)
        start_row = 0

    with open(generations_filename, 'r', encoding="utf-8") as f:
        for num_requests, line in enumerate(f):
            # if num_requests % 10 == 0:
            #    time.sleep(0.2)
            if num_requests == 50000:
                break

            if num_requests < start_row:
                continue

            if num_requests % 500 == 0:
                print("{} | Request #{}".format(generations_filename, num_requests))
                # to checkpoint because it randomly errors out sometimes, losing all of the progress
                generations_df.to_json(scores_filename, orient="records", force_ascii=False, lines=True)

            if line[-1] == "\n":
                line = line[:-1]

            data_dict = {"comment": {"text": line},
                         "languages": ["en"],
                         "requestedAttributes": {"TOXICITY": {}, "SEVERE_TOXICITY": {}, "INSULT": {}, \
                                                 "FLIRTATION": {}, "PROFANITY": {}, "SEXUALLY_EXPLICIT": {}, \
                                                 "THREAT": {}, "IDENTITY_ATTACK": {}}}

            if len(line) == 0:
                row = [""] + [0] * 8
                generations_df.loc[len(generations_df), :] = row
            else:
                response = requests.post(url=url, data=json.dumps(data_dict))
                response_dict = json.loads(response.content)
                if "attributeScores" in response_dict:
                    category_scores = []
                    for categ in categories:
                        score = response_dict["attributeScores"][categ.upper()]["spanScores"][0]["score"]["value"]
                        category_scores.append(score)
                    row = [line] + category_scores
                    generations_df.loc[len(generations_df), :] = row
                else:
                    print(response_dict)

    generations_df.to_json(scores_filename, orient="records", force_ascii=False, lines=True)


if __name__ == '__main__':

  score_generations("./data/GabHate/implicit.txt", r"C:\Users\123\Desktop\GabHateCorpus\perspective\implicit.json")
