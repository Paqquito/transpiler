import json
from transformers import AutoTokenizer
import torch
from pathlib import Path

def tokenize_data(json_folder, output_folder, model_name="EleutherAI/gpt-neo-2.7B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # S'assurer que le tokenizer a un token de padding
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for json_file in Path(json_folder).glob('*.json'):
        with open(json_file, 'r', encoding='utf-8') as file:
            data_pair = json.load(file)
            rte_code = data_pair["rte_code"]
            c_code = data_pair["c_code"]

            inputs = tokenizer(rte_code, padding=True, truncation=True, return_tensors="pt")
            targets = tokenizer(c_code, padding=True, truncation=True, return_tensors="pt")

            torch.save(inputs, Path(output_folder) / f"{json_file.stem}_inputs.pt")
            torch.save(targets, Path(output_folder) / f"{json_file.stem}_targets.pt")

            print(f"Processed {json_file.name}")

if __name__ == "__main__":
    json_folder = 'transpiler/transpile_project/paire_JSON'
    output_folder = 'transpiler/transpile_project/tokenized_data'
    tokenize_data(json_folder, output_folder)
