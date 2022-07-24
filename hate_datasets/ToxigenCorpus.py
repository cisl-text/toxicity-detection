#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   GabHateCorpus.py    
@Contact :   1720613556@qq.com
@License :   (C)Copyright 2021-2022

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/7/4 18:39   dst      1.0         None
'''
import torch
from torch.utils.data import Dataset
from utils import load_data

class ToxigenCorpus(Dataset):
    def __init__(self, tokenizer, data_dir="./data/Toxigen/", mode=5, prepared_data=None,export=False):
        """
        MODE:
        4. TO(IM)
        5. TO + NON
        """
        assert mode>=4 and mode<=5
        self.export = export
        self.tokenizer = tokenizer
        if prepared_data:
            self.data = prepared_data
        else:
            self.switch_mode(data_dir, mode)


    def switch_mode(self, data_dir, mode):
        # implicit
        to_data = self.load_data(data_dir + 'toxic.txt', mode="toxic")
        # nones
        non_data = self.load_data(data_dir + 'non_toxic.txt', mode="none")

        if mode == 4:
            self.data = to_data 
        else:
            self.data = to_data + non_data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label, implicit = self.data[idx]
        return text, label, implicit

    def load_data(self, file_name, mode):
        res_data = []
        with open(file_name, 'r', encoding="utf-8") as f:
            for line in f.readlines()[:13000]:
                # todo: 需要处理文本
                text = line.strip()
                if len(text)==0:
                    continue
                if mode == 'none':
                    # (text, toxic, implicit)
                    res_data.append((text, 0, -1))
                elif mode == "toxic":
                    # toxigen 生成的算隐式吧
                    res_data.append((text, 1, 1))
                else:  # explicit
                    res_data.append((text, 1, 0))
        return res_data

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
            return input_ids, attention_mask,  toxic_labels, implicit_labels, sents


