#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Contact :   1720613556@qq.com
@License :   (C)Copyright 2021-2022

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/7/4 20:09   dst      1.0         None
'''
from cProfile import label
import logging
import os
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from pprint import pprint
import yaml


def evaluate(labels_all, predict_all, POS_LABEL="toxic", eval_all=False, prob_all=None, implicit_label=None):
    """evaluation"""
    acc = accuracy_score(labels_all, predict_all)
    print("Acc:", acc)
    if eval_all:
        assert prob_all is not None
        target_names = ['non-toxic', 'toxic'] if POS_LABEL == "toxic" else ['toxic', 'non-toxic']
        report = classification_report(labels_all, predict_all, target_names=target_names)
        print(report)
        auc = roc_auc_score(labels_all, prob_all)
        print("Auc:", auc)
        implicit_ac=0
        implicit_all=0
        explicit_ac=0
        explicit_all=0
        non_ac=0
        non_all=0
        for i in range(len(labels_all)):
            if implicit_label[i] == 0: 
                explicit_all+=1
                explicit_ac += 1 if labels_all[i]==predict_all[i] else 0
            if implicit_label[i] == 1: 
                implicit_all+=1
                implicit_ac += 1 if labels_all[i]==predict_all[i] else 0
            if implicit_label[i] == -1: 
                non_all+=1
                non_ac += 1 if labels_all[i]==predict_all[i] else 0
        each_acc_str = f"implicit Acc: {float(implicit_ac)/float(implicit_all)}\n"+f"explicit Acc: {float(explicit_ac)/float(explicit_all)}\n"+f"none toxic Acc: {float(non_ac)/float(non_all)}\n"
        return acc, report, auc, each_acc_str
    return acc


def get_config(config_name, mode="train"):
    with open(config_name, 'r') as file:
        try:
            print(f"mode:{mode}")
            params = yaml.safe_load(file)[mode] 
            # 优雅！
            pprint(params, indent=4, sort_dicts=False)
            return params
        except yaml.YAMLError as exc:
            print(exc)


def load_data(file_name, mode):
    res_data = []
    with open(file_name, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            # todo: 需要处理文本
            text = line.strip()
            if mode == 'none':
                # (text, toxic, implicit)
                res_data.append((text, 0, -1))
            elif mode == "implicit":
                res_data.append((text, 1, 1))
            else:  # explicit
                res_data.append((text, 1, 0))
    return res_data


def split_data(data_dir="./data/ImplicitHate/", split_ratio=0.8, shuffle=False):
    # implicit
    implicit_data = load_data(data_dir + 'implicit.txt', mode="implicit")
    # explicit
    explicit_data = load_data(data_dir + 'explicit.txt', mode="explicit")
    # non
    non_data = load_data(data_dir + 'non_toxic.txt', mode="none")
    if shuffle:
        shuffle(implicit_data), shuffle(explicit_data), shuffle(non_data)

    def split_data(l, ratio):
        split_point = int(len(l) * ratio)
        return l[:split_point], l[split_point:]

    train_implicit_data, test_implicit_data = split_data(implicit_data, split_ratio)
    train_explicit_data, test_explicit_data = split_data(explicit_data, split_ratio)
    train_non_data, test_non_data = split_data(non_data, split_ratio)

    train_data = train_implicit_data + train_explicit_data + train_non_data
    test_data = test_implicit_data + test_explicit_data + test_non_data
    return train_data, test_data

import spacy
nlp = spacy.load("en_core_web_sm")
def dependency_adj_matrix(text):
    tokens = nlp(text)
    tokens_text = []
    matrix = np.zeros((len(tokens)+1, len(tokens)+1)).astype('float32')
    # 句法分析空出[CLS]
    for token in tokens:
        tokens_text.append(token.text)
        matrix[token.i+1][token.i+1] = 1
        for child in token.children:
            matrix[token.i+1][child.i+1] = 1
            matrix[child.i+1][token.i+1] = 1
    return matrix, tokens_text


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = -np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        # set logger
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
        self.logger = self.get_logger(os.path.join("./log", time_str + ".log"))

    def get_logger(self, filename, verbosity=1, name=None):
        level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
        formatter = logging.Formatter(
            "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
        )
        logger = logging.getLogger(name)
        logger.setLevel(level_dict[verbosity])

        fh = logging.FileHandler(filename, "w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # 控制台输出
        # sh = logging.StreamHandler()
        # sh.setFormatter(formatter)
        # logger.addHandler(sh)
        return logger

    def __call__(self, val_acc, model, progress_summary, epoch, optimizer, loss, scheduler ):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model, epoch, optimizer, loss, scheduler)
            self.logger.info(progress_summary)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # todo: 打印最好的一次的结果（时间）
            self.best_score = score
            self.save_checkpoint(val_acc, model, epoch, optimizer, loss, scheduler)
            self.counter = 0
            self.logger.info(progress_summary)

    def save_checkpoint(self, val_acc, model, epoch, optimizer, loss, scheduler):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')

        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': loss,
        #     'scheduler':scheduler,
        #     }, os.path.join(self.path,f"multiview{time_str}.pth"))
        torch.save(model.state_dict(), os.path.join(self.path,f"multiview{time_str}.pth"))
        self.logger.info(f"g:{model.g}")
        # model.save_pretrained(self.path)
        self.val_acc_max = val_acc


def make_bert_input(sent_tokens, tokenizer, pad_len = 500):
    input_all = []

    for tokens in sent_tokens:
        input_id =[]
        input_id.extend(tokenizer.convert_tokens_to_ids(tokens))
        input_one = tokenizer.encode_plus(input_id, padding='max_length', max_length=pad_len)
        input_all.append(input_one)
    input_ids = torch.Tensor([item['input_ids'] for item in input_all]).long()
    att_mask = torch.Tensor([item['attention_mask'] for item in input_all]).long()
    return input_ids, att_mask