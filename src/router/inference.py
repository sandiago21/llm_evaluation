import yaml
import torch
import numpy as np

from src.router.model import load_router_model, tokenizer

MODEL_NAMES = [
    "mistral",
    "gemma2",
]

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)


def predict_best_model(query: str):

    model = load_router_model()

    max_score = 0
    recommended_model = ""

    for model_name in MODEL_NAMES:
        word = torch.tensor([len(query.split())])
        character = torch.tensor([len(query)])

        encoded_input = query + "[SEP]" + model_name

        encoded = tokenizer(
                [encoded_input],
                padding = 'max_length',            
                max_length = config["transformer_model"]["max_len"],
                truncation = True,
                return_attention_mask=True
            )

        input_ids = torch.tensor(encoded['input_ids'])
        attention_mask = torch.tensor(encoded['attention_mask'])

        with torch.no_grad():
            input_ids = input_ids.to(config["transformer_model"]["DEVICE"])
            attention_mask = attention_mask.to(config["transformer_model"]["DEVICE"])
            word = word.to(config["transformer_model"]["DEVICE"])
            character = character.to(config["transformer_model"]["DEVICE"])
                        
            pred = model(input_ids, attention_mask, word, character).flatten().to("cpu")
            score = float(pred.numpy()[0])

        if score > max_score:
            max_score = score
            recommended_model = model_name

    return {
        "recommended_model": recommended_model,
        "confidence": max_score,
    }
