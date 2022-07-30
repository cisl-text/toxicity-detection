python3 toxicity_augment.py \
        --model_name_or_path /dev-data/ybshu/plms/gpt-neo-1.3B \
        --source_file /dev-data/ybshu/code/research/toxicity-detection/data/ImplicitHate/explicit.txt \
        --start_idx 0 \
        --end_idx 20000 \
        --target_file /dev-data/ybshu/code/research/toxicity-detection/data/ImplicitHate/aug_hate/explicit \
        --length 128 \
        --prompt "I hate" \
        --device cuda:1