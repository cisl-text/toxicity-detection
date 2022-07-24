import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import get_full_repo_name
from transformers.utils.versions import require_version
from utils import evaluate, get_config, EarlyStopping, split_data
from hate_datasets.GabHateCorpus import GabHateCorpus
from hate_datasets.ImplicitHateCorpus import ImplicitHateCorpus
from hate_datasets.SBIC import SBICDataset
from hate_datasets.ToxigenCorpus import ToxigenCorpus
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "GabHate": ("sentence", None),
    "ImplicitHate": ("sentence", None),
    "SBIC": ("sentence", None),
    "Toxigen": ("sentence", None)
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=False,
        help="whether or not to shuffle datasets"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="the number of workers to acquire data batches"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def get_label_list(dataset, choice=1):
    """
    Get the list of labels from the dataset.
    """
    data = dataset.data
    labels = set([triplet[choice] for triplet in data])
    label_list = sorted(list(labels))
    return label_list

def prepare_dataset(task_name, tokenizer, splitRatio=0.2, shuffle=True):
    if task_name == "GabHate":
        train_data, test_data = split_data(data_dir="./data/GabHate/", split_ratio=splitRatio,
                                                            shuffle=shuffle)
        train_dataset = GabHateCorpus(prepared_data=train_data, tokenizer=tokenizer)
        test_dataset = GabHateCorpus(prepared_data=test_data, tokenizer=tokenizer)
    elif task_name == 'ImplicitHate':
        train_data, test_data = split_data(data_dir="./data/ImplicitHate/",split_ratio=splitRatio,shuffle=shuffle)
        train_dataset = ImplicitHateCorpus(prepared_data=train_data, tokenizer=tokenizer)
        test_dataset = ImplicitHateCorpus(prepared_data=test_data, tokenizer=tokenizer)
    return train_dataset, test_dataset

def finetune():

    # 1. arguments
    args = parse_args()

    # 2. init accelerator
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir= args.output_dir) 
        if args.with_tracking else Accelerator() )
    # for log
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # 3. prepare dataset corresponding to the task
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    train_dataset, test_dataset = prepare_dataset(args.task_name, tokenizer)
    label_list = get_label_list(train_dataset)
    num_label = len(label_list)

    # 4. prepare model
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_label, finetuning_task=args.task_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )
    
    # 5. data collator, data loader
    train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=args.shuffle, 
                        num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.per_device_eval_batch_size, shuffle=args.shuffle, 
                        num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
    
    # 6. prepare optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)


    # 7. scheduler
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader)/args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # 8. prepare everything in accelerator
    model, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 9. train
    #metric = load_metric("accuracy")
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    for epoch in range(starting_epoch, args.num_train_epochs):

        # train loop
        model.train()
        if args.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            inputs = {
                "input_ids":batch[0],
                "attention_mask":batch[1],
                "labels":batch[2]}
            outputs = model(**inputs)
            loss = outputs.loss
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss/args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step%args.gradient_accumulation_steps==0 or step==len(train_dataloader)-1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            
            if completed_steps >= args.max_train_steps:
                break

        # evluation
        model.eval()
        samples_seen = 0
        test_predictions = []
        test_references = []
        for step, batch in enumerate(test_dataloader):
            with torch.no_grad():
                inputs = {
                    "input_ids":batch[0],
                    "attention_mask":batch[1],
                    "labels":batch[2]}
                outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).squeeze()
            predictions, references = accelerator.gather((predictions, inputs["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(test_dataloader) - 1:
                    predictions = predictions[:len(test_dataloader)-samples_seen]
                    references = references[:len(test_dataloader)-samples_seen]
                else:
                    samples_seen += references.shape[0]
            test_predictions += predictions
            test_references += references
        
        acc = evaluate(test_references, test_predictions, "toxic", eval_all=False)
        print(epoch, acc)
            



if __name__ == "__main__":
    finetune()



















