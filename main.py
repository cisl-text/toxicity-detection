import torch
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
device_num = 4
import warnings
from model_code.KGNN import KGNN
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from utils import get_config, split_data,EarlyStopping,evaluate
from transformers import AutoTokenizer, AdamW
import transformers
import time
from hate_datasets.ImplicitHateCorpus import ImplicitHateCorpus
class ModelTrainer:
    def __init__(self, train_config, model_config):
        self.train_config = train_config
        self.tokenizer = AutoTokenizer.from_pretrained(train_config['dataset']['tokenizer'])
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # finetune data: train, test
        train_dataset, test_dataset = self.prepare_dataset()
        self.train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True,
                                           num_workers=train_config['dataset']['num_workers'],
                                           collate_fn=train_dataset.collate_fn)
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False,
                                          num_workers=train_config['dataset']['num_workers'],
                                          collate_fn=test_dataset.collate_fn)

        self.device_ids = [i for i in range(device_num)]
        self.model = KGNN(bert_model=model_config['bert_model'], word_emb_dim=model_config['word_emb_dim'],
                            lstm_hid_dim=model_config['lstm_hid_dim'], dropout_rate=model_config['dropout_rate'])
        self.model = torch.nn.DataParallel(self.model,device_ids=[1,2,3,4,5,6,7])
        self.model = self.model.to(self.device)
        # loss
        self.loss = CrossEntropyLoss()
        # optimizer & scheduler
        self.get_optimizer()
        self.optimizer =torch.nn.DataParallel(self.optimizer,device_ids=[1,2,3,4,5,6,7])
        self.scheduler = torch.nn.DataParallel(self.scheduler,device_ids=[1,2,3,4,5,6,7])
        # save
        self.save_path = train_config['saveDir']
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.earlystop = EarlyStopping(patience=train_config['patience'], verbose=False, delta=train_config['delta'],
                                       path=self.save_path, trace_func=print)

    def prepare_dataset(self):
        if self.train_config['dataset']['name'] == 'ImplicitHateCorpus':
            train_data, test_data = split_data(data_dir="./data/ImplicitHate/",split_ratio=self.train_config['dataset']['splitRatio'],shuffle=self.train_config['dataset']['shuffle'])
            train_dataset = ImplicitHateCorpus(prepared_data=train_data, tokenizer=self.tokenizer, add_dep=self.train_config['dataset']['add_dep'])
            test_dataset = ImplicitHateCorpus(prepared_data=test_data, tokenizer=self.tokenizer, add_dep=self.train_config['dataset']['add_dep'])
        else:
            pass
        return train_dataset, test_dataset

    def get_optimizer(self):
        if self.train_config['weight_decay']:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.train_config['lr'],
                                   correct_bias=not self.train_config['bertadam'])

        else:
            self.optimizer = AdamW(self.model.parameters(), lr=self.train_config['lr'], betas=(0.9, 0.999),
                                   eps=1e-08, weight_decay=0.01, correct_bias=not self.train_config['bertadam'])
        num_train_optimization_steps = len(self.train_dataloader) * self.train_config['epoch']
        if self.train_config['warmup_proportion'] != 0:
            self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                          int(num_train_optimization_steps * self.train_config[
                                                                              'warmup_proportion']),
                                                                          num_train_optimization_steps)
        else:
            self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                          int(num_train_optimization_steps * self.train_config[
                                                                              'warmup_proportion']),
                                                                          num_train_optimization_steps)


    def finetune(self):
        loss_total = [0]
        for epoch in range(self.train_config['epoch']):
            self.model.train()
            start_time = time.time()

            tqdm_bar = tqdm(self.train_dataloader, desc=f"Training epoch{epoch}, mean loss{np.mean(loss_total)}",
                            total=len(self.train_dataloader))
            for i, (input_ids, attention_mask, labels, implicit_labels, adj_matrix) in enumerate(tqdm_bar):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                adj_matrix = adj_matrix.to(self.device)
                self.model.module.zero_grad()
                out = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask, adj=adj_matrix)
                if i==10:
                    print(out)
                loss = self.loss(out, labels)
                loss.backward()
                self.optimizer.module.step()
                self.scheduler.module.step()
                self.optimizer.module.zero_grad()
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
        for i, (input_ids, attention_mask, labels, implicit_labels, adj_matrix) in enumerate(self.test_dataloader):
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            adj_matrix = adj_matrix.to(self.device)
            with torch.no_grad():
                out = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask, adj=adj_matrix)
            out = out.cpu()
            total_loss += self.loss(out, labels).item()
            pred_prob_all.extend(out[:, 1])
            out = out.argmax(dim=1)
            label_all.extend(labels)
            pred_all.extend(out)
        acc, report, auc = evaluate(label_all, pred_all, "toxic", eval_all=True,
                                    prob_all=pred_prob_all)
        progress_summary = f"Epoch {epoch + 1}: loss:{total_loss / len(self.test_dataloader)} acc:{acc}, auc: {auc} \n" + report
        self.earlystop(acc, self.model, progress_summary)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='bert classification')
    parser.add_argument("-c", "--config", type=str, default="./configs/multi_view_config.yaml")
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    # train
    train_config = get_config(args.config, mode="train")
    # os.environ["CUDA_VISIBLE_DEVICES"] = '4'

    
    model_config = get_config(args.config, mode="model")
    trainer = ModelTrainer(train_config, model_config)
    trainer.finetune()
    
