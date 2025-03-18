import argparse
import torch
import os
import json
from transformers import AutoTokenizer, LlamaForCausalLM
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset-file', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    return args


def get_dataset(dataset_file):
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)

    output_dataset = dict()
    for task, task_dataset in dataset.items():
        output_dataset[task] = dict()
        for datum_id, datum in task_dataset.items():
            text = "<|begin_of_text|>"
            suffix = "<|eot_id|>"
            for message in datum:
                if message["role"] == "user":
                    text += "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>".format(message["text"])
                elif message["role"] == "system":
                    text += "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>".format(message["text"])
                elif message["role"] == "assistant":
                    text += "<|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>".format(message["text"])

            assert message["role"] == "assistant"
            last_round_text = message["text"]
            local_offset = 0
            for ref in message["evaluation_reference"]:
                if ref in last_round_text:
                    local_offset = last_round_text.find(ref)
                    break
            offset = len(text) - len(last_round_text) - len(suffix)\
                + local_offset
            # print(len(text), len(last_round_text), offset, local_offset)

            output_dataset[task][datum_id] = {
                "text": text,
                "evaluation_reference": message["evaluation_reference"],
                "last_round_text": last_round_text,
                "offset": offset,
                "local_offset": local_offset,
            }
    return output_dataset


if __name__ == "__main__":
    random.seed(0)
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    dataset = get_dataset(args.dataset_file)
    # load llama
    model = LlamaForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    n_layers = model.config.num_hidden_layers
    model.eval()

    all_indices = [
        (task, datum_id)
        for task, task_dataset in dataset.items()
        for datum_id in task_dataset.keys()
    ]
    random.shuffle(all_indices)

    for task, datum_id in tqdm(all_indices[:args.num_samples]):
        save_file = os.path.join(args.output_dir, f"{task}_{datum_id}.pdf")
        datum = dataset[task][datum_id]
        text = datum["text"]
        tokenized = tokenizer(text, return_tensors="pt",
                              return_offsets_mapping=True)
        token_length = tokenized.input_ids.size(1)

        if token_length > 2048:
            print(f"Skip {task}_{datum_id} due to long text")
            continue
        if os.path.exists(save_file):
            print(f"Skip {task}_{datum_id} due to existing file")
            continue

        offset = len([x for x in tokenized["offset_mapping"][0]
                      if x[1] < datum["offset"]])
        output = model(
            tokenized.input_ids[:, :offset],
            attention_mask=tokenized.attention_mask[:, :offset],
            output_attentions=True,
        )
        attentions = output.attentions
        spot_attention = [
            layer_attention[0, :, -1, :]
            for layer_attention in attentions
        ]

        fig, axes = plt.subplots(n_layers, 1, figsize=(10, 10))
        for i, ax in enumerate(axes):
            ax.matshow(spot_attention[i].detach().numpy(), cmap="hot",
                       interpolation="nearest", aspect="auto")
            ax.axis("off")
        # draw all tokens along x-axis using a small font with ax.text,
        ax = axes[0]
        for j, token in enumerate(tokenized["input_ids"][0, :offset]):
            ax.text(j, -2, tokenizer.decode(token),
                    ha="center", va="center", fontsize=0.0001,
                    rotation=90)
        plt.tight_layout()
        margin = 0.1
        plt.subplots_adjust(
            left=margin, right=1-margin,
            top=1-margin, bottom=margin,
            wspace=margin, hspace=margin)
        plt.savefig(save_file)
        print(f"Saved to {save_file}")
