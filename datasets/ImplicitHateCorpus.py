#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ImplicitHateCorpus.py    
@Contact :   1720613556@qq.com
@License :   (C)Copyright 2021-2022
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/7/5 17:03   dst      1.0         None
'''

import torch
from torch.utils.data import Dataset
from utils import load_data

class ImplicitHateCorpus(Dataset):
    def __init__(self, tokenizer, data_dir="./data/ImplicitHate/", mode=5, prepared_data=None, export=False):
        """
        MODE:
        1. IM
        2. EX
        3. None
        4. EX + IM
        5. EX +IM + NON
        """
        self.export = export
        self.tokenizer = tokenizer

        if prepared_data:
            self.data = prepared_data
        else:
            self.switch_mode(data_dir, mode)

    def switch_mode(self, data_dir, mode):
        # implicit
        implicit_data = load_data(data_dir + 'implicit.txt', mode="implicit")
        # explicit
        explicit_data = load_data(data_dir + 'explicit.txt', mode="explicit")
        # non
        non_data = load_data(data_dir + 'non_toxic.txt', mode="none")
        # mode
        if mode == 1:
            self.data = implicit_data
        elif mode == 2:
            self.data = explicit_data
        elif mode == 3:
            self.data = non_data
        elif mode == 4:
            self.data = implicit_data + explicit_data
        else:
            self.data = implicit_data + explicit_data + non_data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label, implicit = self.data[idx]
        return text, label, implicit

    def collate_fn(self, data):
        sents = [i[0] for i in data]
        toxic_labels = [i[1] for i in data]
        implicit_labels = [i[2] for i in data]

        # 编码
        data = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                                truncation=True,
                                                padding='max_length',
                                                max_length=500,
                                                return_tensors='pt',
                                                return_length=True)

        # input_ids:编码之后的数字
        # attention_mask:是补零的位置是0,其他位置是1
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        toxic_labels = torch.LongTensor(toxic_labels)
        implicit_labels = torch.LongTensor(implicit_labels)
        if self.export == False:
            return input_ids, attention_mask,  toxic_labels, implicit_labels
        else:
            return input_ids, attention_mask,  toxic_labels, implicit_labels,