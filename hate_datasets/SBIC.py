#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   SBIC.py    
@Contact :   1720613556@qq.com
@License :   (C)Copyright 2021-2022

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/7/18 9:37   dst      1.0         None
'''
import os
import pickle

import torch
from torch.utils.data import Dataset
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, note="", micro=None):
        """Constructs a InputExample.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.note = note
        self.micro = micro

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, guid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.guid = guid

class MAProcessor(object):

    def get_direct_control_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "micro_train.pkl")),
                                     "missed_micro") + self._create_examples(
            self._read_tsv(os.path.join(data_dir, "nonmicro_large_train.pkl")), "clean") + self._create_examples(
            self._read_tsv(os.path.join(data_dir, "hs_train.pkl")), "hateful")

    def get_dirctr_corrected_train_examples(self, data_dir, correction_dir, correction_size):
        """See base class."""
        sorted_ex_for_correction = self._read_tsv(os.path.join(correction_dir, "sorted_ex_for_correction.pkl"))
        correct_idx_list = sorted_ex_for_correction[:correction_size]
        examples = self._create_examples(self._read_tsv(os.path.join(data_dir, "micro_train.pkl")),
                                         "missed_micro") + self._create_examples(
            self._read_tsv(os.path.join(data_dir, "nonmicro_large_train.pkl")), "clean") + self._create_examples(
            self._read_tsv(os.path.join(data_dir, "hs_train.pkl")), "hateful")
        new_examples = []
        for i, example in enumerate(examples):
            if i in correct_idx_list:
                new_examples.append(
                    InputExample(guid=i, text_a=example.text_a, text_b=None, label="1"))  # flip the label
            else:
                new_examples.append(InputExample(guid=i, text_a=example.text_a, text_b=None, label=example.label))
        return new_examples

    def get_dirctr_checked_train_examples(self, data_dir, correction_dir, correction_size):
        """See base class."""
        sorted_ex_for_correction = self._read_tsv(os.path.join(correction_dir, "sorted_ex_for_correction.pkl"))
        correct_idx_list = sorted_ex_for_correction[:correction_size]
        len_micro_train = len(self._read_tsv(os.path.join(data_dir, "micro_train.pkl")))
        examples = self._create_examples(self._read_tsv(os.path.join(data_dir, "micro_train.pkl")),
                                         "missed_micro") + self._create_examples(
            self._read_tsv(os.path.join(data_dir, "nonmicro_large_train.pkl")), "clean") + self._create_examples(
            self._read_tsv(os.path.join(data_dir, "hs_train.pkl")), "hateful")
        new_examples = []
        for i, example in enumerate(examples):
            if i in correct_idx_list and i < len_micro_train:  # only flip the true microaggressions
                new_examples.append(
                    InputExample(guid=i, text_a=example.text_a, text_b=None, label="1"))  # flip the label
            else:
                new_examples.append(InputExample(guid=i, text_a=example.text_a, text_b=None, label=example.label))
        return new_examples

    def get_dirctr_gold_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "micro_train.pkl")),
                                     "gold_micro") + self._create_examples(
            self._read_tsv(os.path.join(data_dir, "nonmicro_large_train.pkl")), "clean") + self._create_examples(
            self._read_tsv(os.path.join(data_dir, "hs_train.pkl")), "hateful")

    def get_direct_control_test_examples(self, data_dir):
        """See base class."""
        micro_test_ex = self._create_examples(self._read_tsv(os.path.join(data_dir, "micro_test.pkl")), "gold_micro")
        clean_test_ex = self._create_examples(self._read_tsv(os.path.join(data_dir, "nonmicro_large_test.pkl")),
                                              "clean")
        hs_test_ex = self._create_examples(self._read_tsv(os.path.join(data_dir, "hs_test.pkl")), "hateful")
        return micro_test_ex + clean_test_ex + hs_test_ex

    def get_adv_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "micro_adv.pkl")), "missed_micro")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text_a = line[0]
            if set_type == "hateful":
                label = "1"
                micro = "0"
            elif set_type == "missed_micro":
                label = "0"
                micro = "1"
            elif set_type == "gold_micro":
                label = "1"
                micro = "1"
            elif set_type == "clean":
                label = "0"
                micro = "-1"
            else:
                raise ValueError("Check your set type")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, micro=micro))
        return examples

    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "rb") as f:
            pairs = pickle.load(f)
            return pairs

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    # todo: 没有使用这种convert
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          guid=example.guid))
    return features

class SBICDataset(Dataset):
    def __init__(self, tokenizer, data_dir="./data/SBIC/processed_dataset", mode='dirctr_gold_train', prepared_data=None, export=False, correction_dir=None,correction_size=None):
        """
        dirctr_missed_train：将implicit 的数据标为0 miss micro
        dirctr_corrected_train： 更正错误的样例
        dirctr_checked_train：更正implicit错误的样例
        dirctr_gold_train：完全正确的原始数据

        :param tokenizer:
        :param data_dir:
        :param mode:
        :param prepared_data:
        :param export:
        """
        self.export = export
        self.tokenizer = tokenizer

        if prepared_data:
            self.data = prepared_data
        else:
            ma_processor = MAProcessor()
            label_list = ma_processor.get_labels()
            if mode == "dirctr_missed_train":
                self.data = ma_processor.get_direct_control_train_examples(data_dir)
            elif mode == "dirctr_corrected_train":

                assert (correction_dir is not None) and (correction_size is not None)
                self.data = ma_processor.get_dirctr_corrected_train_examples(data_dir, correction_dir,correction_size)
            elif mode == "dirctr_checked_train":
                assert (correction_dir is not None) and (correction_size is not None)
                self.data = ma_processor.get_dirctr_checked_train_examples(data_dir, correction_dir, correction_size)
            elif mode == "dirctr_gold_train":
                self.data = ma_processor.get_dirctr_gold_train_examples(data_dir)
            elif mode == "dirctr_adv":
                self.data = ma_processor.get_adv_examples(data_dir)
            elif mode == "dirctr_test":
                self.data = ma_processor.get_direct_control_test_examples(data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return  self.data[idx]

    def collate_fn(self, data):

        sents = [d.text_a for d in data]
        toxic_labels = [int(d.label) for d in data]
        implicit_labels = [int(d.micro) for d in data]

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
            return input_ids, attention_mask, toxic_labels, implicit_labels
        else:
            return input_ids, attention_mask, toxic_labels, implicit_labels, sents