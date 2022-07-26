import argparse
import logging
import numpy as np
import torch

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPTNeoForCausalLM,
    AutoTokenizer
)

from csv import DictWriter
from tqdm import tqdm

class Writer():
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device if device==None else device
    
    def write_one(self, prefix, prompt = "I think", length = 128):
        encoded_prompt = self.tokenizer.encode(" ".join([prefix, prompt]), add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.model.device)

        input_ids = encoded_prompt
        eos_token = self.tokenizer.eos_token
        eos_token_id = self.tokenizer.eos_token_id

        # simple config
        if True:
            length = length
            temperature = 1
            num_beam = 5
            k = 50
            p = 1
            repetition_penalty = 1
            do_sample=True
            num_return_sequence = 3

        output_sequences = self.model.generate(
            input_ids = input_ids,
            max_length = len(encoded_prompt[0]) + length,
            temperature = temperature,
            #num_beam = num_beam,
            top_k = k,
            top_p = p,
            repetition_penalty = repetition_penalty,
            do_sample = do_sample,
            num_return_sequence = num_return_sequence,
            pad_token_id = eos_token_id
        )


        generated_sequences = []
        stop_token = self.tokenizer.eos_token
        # stop_token = "\n"

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):

            generated_sequence = generated_sequence.tolist()
            # Decode text
            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
            # Remove all text after the stop token
            text = text[: text.find(stop_token) if stop_token else None]
            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                prompt + text[len(self.tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
            )

            generated_sequences.append(total_sequence)
            #print(total_sequence)
        return generated_sequences


    def write_and_save(self, prefix_source, target, start_idx, end_idx, prompt = "I think", length = 128):
        
        f = open(prefix_source, "r")
        field_names = ['prefix','prompt','generation']
        target = target + f"_{start_idx}_{end_idx}" + ".csv"

        idx = -1
        print("start:", start_idx, "end:", end_idx)
        for line in tqdm(f.readlines()):
            idx+=1
            if start_idx<=idx<=end_idx:
                prefix = line.strip()
                generation = self.write_one(prefix, prompt, length)
                if len(generation)==0:
                    continue
                row = {"prefix":prefix, "prompt":prompt, "generation":generation[0]}
                with open(target, "a") as out_f:
                    dictwriter_object = DictWriter(out_f, fieldnames=field_names)
                    dictwriter_object.writerow(row)

        f.close()
        print("Done!")
                


def parse_args():
    parser = argparse.ArgumentParser(description="Running transformers model on text generation")

    parser.add_argument(
        "--model_name_or_path", type=str, default="/dev-data/ybshu/plms/gpt-neo-125M"
    )
    parser.add_argument(
        "--source_file", type=str, default=None
    )
    parser.add_argument(
        "--start_idx", type=int, default=0
    )
    parser.add_argument(
        "--end_idx", type=int, default=-1
    )
    parser.add_argument(
        "--target_file", type=str, default=None
    )
    parser.add_argument(
        "--length",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--prompt", type=str, default="I think"
    )
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    return args



def main():
    # init
    args = parse_args()
    model_name_or_path = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = GPTNeoForCausalLM.from_pretrained(model_name_or_path)
    model = model.to(args.device)
    writer = Writer(model, tokenizer)

    # write
    writer.write_and_save(prefix_source = args.source_file, 
                        target = args.target_file,
                        start_idx = args.start_idx,
                        end_idx = args.end_idx,
                        prompt = args.prompt,
                        length = args.length)

if __name__ == "__main__":
    main()
