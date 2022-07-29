#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   finetune_bert.py
@Contact :   1720613556@qq.com
@License :   (C)Copyright 2021-2022

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/7/7 17:30   dst      1.0         None
'''
import argparse
import os
import time

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, BertForSequenceClassification, RobertaForSequenceClassification
from utils import evaluate, get_config, EarlyStopping, split_data
import transformers
from datasets.GabHateCorpus import GabHateCorpus
from datasets.ImplicitHateCorpus import ImplicitHateCorpus
from datasets.SBIC import SBICDataset, SBICMTLDataset


# 参考：https://zhuanlan.zhihu.com/p/524036087
class BertTrainer:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # finetune data: train, test
        train_dataset, test_dataset = self.prepare_dataset()
        self.train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                                           num_workers=config['num_workers'],
                                           collate_fn=train_dataset.collate_fn)
        self.test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False,
                                          num_workers=config['num_workers'],
                                          collate_fn=test_dataset.collate_fn)

        # AutoModelForSequenceClassification.from_pretrained(r"C:\Users\17206\Desktop\save", num_labels=5, problem_type="multi_label_classification", ignore_mismatched_sizes=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(config['model']).to(self.device)
        self.init_parameters()
        # loss
        self.loss = CrossEntropyLoss()
        # optimizer & scheduler
        self.get_optimizer()
        self.eval_all = config['evalAll']
        # save
        self.save_path = config['saveDir']
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.earlystop = EarlyStopping(patience=config['patience'], verbose=False, delta=config['delta'],
                                       path=self.save_path, trace_func=print, savePretrained=config['savePretrained'])

    def prepare_dataset(self):
        if self.config['dataset']['name'] == "GabHateCorpus":
            train_data, test_data = split_data(data_dir="./data/GabHate/", split_ratio=self.config['dataset']['splitRatio'],
                                                             shuffle=self.config['dataset']['shuffle'])
            train_dataset = GabHateCorpus(prepared_data=train_data, tokenizer=self.tokenizer)
            test_dataset = GabHateCorpus(prepared_data=test_data, tokenizer=self.tokenizer)
        elif self.config['dataset']['name'] == 'ImplicitHateCorpus':
            train_data, test_data = split_data(data_dir="./data/ImplicitHate/",split_ratio=self.config['dataset']['splitRatio'],shuffle=self.config['dataset']['shuffle'])
            train_dataset = ImplicitHateCorpus(prepared_data=train_data, tokenizer=self.tokenizer)
            test_dataset = ImplicitHateCorpus(prepared_data=test_data, tokenizer=self.tokenizer)
        elif self.config['dataset']['name'] == 'SBIC':
            if self.config['dataset']['mode'] =='dirctr_gold_train':
                train_dataset = SBICDataset(self.tokenizer, mode='dirctr_gold_train', correction_dir=self.config['dataset']['correction_dir'], correction_size=self.config['dataset']['correction_size'])
                test_dataset = SBICDataset(self.tokenizer, mode="dirctr_test", correction_dir=self.config['dataset']['correction_dir'], correction_size=self.config['dataset']['correction_size'])
            elif self.config['dataset']['mode'] =='lewd':
                train_data, test_data = SBICMTLDataset.get_MTL_data(data="lewd", split_ratio=self.config['dataset']['splitRatio'], shuffle=self.config['dataset']['shuffle'])
                train_dataset = SBICMTLDataset(self.tokenizer, prepared_data=train_data)
                test_dataset = SBICMTLDataset(self.tokenizer, prepared_data=test_data)
            elif self.config['dataset']['mode'] == 'international':
                train_data, test_data = SBICMTLDataset.get_MTL_data(data="international",split_ratio=self.config['dataset']['splitRatio'], shuffle=self.config['dataset']['shuffle'])
                train_dataset = SBICMTLDataset(self.tokenizer, prepared_data=train_data)
                test_dataset = SBICMTLDataset(self.tokenizer, prepared_data=test_data)
            elif self.config['dataset']['mode'] =='targetCategory':
                # train_dataset = SBICMTLDataset(self.tokenizer, prepared_data=train_data)
                # test_dataset = SBICMTLDataset(self.tokenizer, prepared_data=test_data)
                pass # fixme : 多标签多分类任务，之后在补充吧！
        return train_dataset, test_dataset

    def get_optimizer(self):
        if config['weight_decay']:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.config['lr'],
                                   correct_bias=not self.config['bertadam'])

        else:
            self.optimizer = AdamW(self.model.parameters(), lr=self.config['lr'], betas=(0.9, 0.999),
                                   eps=1e-08, weight_decay=0.01, correct_bias=not self.config['bertadam'])
        num_train_optimization_steps = len(self.train_dataloader) * config['epoch']
        if config['warmup_proportion'] != 0:
            self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                          int(num_train_optimization_steps * config[
                                                                              'warmup_proportion']),
                                                                          num_train_optimization_steps)
        else:
            self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                          int(num_train_optimization_steps * config[
                                                                              'warmup_proportion']),
                                                                          num_train_optimization_steps)

    def init_parameters(self):
        if  isinstance(self.model, BertForSequenceClassification):
            self.model.classifier.reset_parameters()  # 使用的是默认初始化 均匀分布
        elif isinstance(self.model, RobertaForSequenceClassification):
            self.model.classifier.dense.reset_parameters()
            self.model.classifier.out_proj.reset_parameters()

    def finetune(self):
        loss_total = [0]
        for epoch in range(config['epoch']):
            self.model.train()
            start_time = time.time()

            tqdm_bar = tqdm(self.train_dataloader, desc=f"Training epoch{epoch}, mean loss{np.mean(loss_total)}",
                            total=len(self.train_dataloader))
            for i, (input_ids, attention_mask, labels, implicit_labels) in enumerate(tqdm_bar):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                self.model.zero_grad()
                out = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask)
                loss = self.loss(out.logits, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                loss = loss.cpu()
                loss_total.append(loss.detach().item())
            self.eval(epoch)
            if self.earlystop.early_stop:
                break

    def eval(self, epoch):
        self.model.eval()
        total_loss = 0
        label_all = []
        pred_all = []
        pred_prob_all = []
        for i, (input_ids, attention_mask, labels, implicit_labels) in enumerate(self.test_dataloader):
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            with torch.no_grad():
                out = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask)

            out = out.logits.cpu()
            total_loss += self.loss(out, labels).detach().item()
            pred_prob_all.extend(out[:, 1])
            out = out.argmax(dim=1)
            # else:
            #     out = out.argmin(dim=1)
            label_all.extend(labels)
            pred_all.extend(out)

        if self.eval_all:
            acc, report, auc = evaluate(label_all, pred_all, config['dataset']['label_names'], eval_all=True,
                                        prob_all=pred_prob_all)
            progress_summary = f"Epoch {epoch + 1}: loss:{total_loss / len(self.test_dataloader)} acc:{acc}, auc: {auc} \n" + report
        else:
            acc = evaluate(label_all, pred_all, config['dataset']['label_names'], eval_all=False)
            progress_summary = f"Epoch {epoch + 1}: loss:{total_loss / len(self.test_dataloader)} acc:{acc} \n"
        self.earlystop(acc, self.model, progress_summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='bert classification')
    parser.add_argument("-c", "--config", type=str, default="./configs/finetune_bert_config.yaml")
    args = parser.parse_args()

    config = get_config(args.config, mode="train")
    os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda']

    trainer = BertTrainer(config)
    trainer.finetune()
