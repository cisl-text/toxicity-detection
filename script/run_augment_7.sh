python3 toxicity_augment.py \
        --model_name_or_path /dev-data/ybshu/plms/gpt-neo-1.3B \
        --source_file /dev-data/ybshu/code/research/toxicity-detection/data/ImplicitHate/non_toxic.txt \
        --start_idx 7000 \
        --end_idx 8000 \
        --target_file /dev-data/ybshu/code/research/toxicity-detection/data/ImplicitHate/augment/non_toxic \
        --length 128 \
        --prompt "I think" \
        --device cuda:10