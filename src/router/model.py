import numpy as np # linear algebra
import yaml
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoConfig
from transformers import get_cosine_schedule_with_warmup

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

tokenizer = AutoTokenizer.from_pretrained(
    config["transformer_model"]["base_transformer_model"],
    use_fast=False,
)

router_model = None

class LLMDataset(Dataset):
    def __init__(self, df, inference_only=False):
        super().__init__()

        self.df = df        
        self.inference_only = inference_only
        self.target = df.correctness_over_latency.astype(float).tolist()
        self.text = [x + "[SEP]" + y for x, y in zip(df["query"], df["model_name"])]
        self.words = df.words.tolist()
        self.characters = df.characters.tolist()
        self.avg_chars_per_word = df.avg_chars_per_word.tolist()

        self.encoded = tokenizer(
            self.text,
            padding = 'max_length',            
            max_length = config["transformer_model"]["max_len"],
            truncation = True,
            return_attention_mask=True
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):        
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        attention_mask = torch.tensor(self.encoded['attention_mask'][index])
        avg_chars_per_word = torch.tensor([self.avg_chars_per_word[index]])
        word = torch.tensor([self.words[index]])
        character = torch.tensor([self.characters[index]])

        if self.inference_only:
            return (input_ids, attention_mask, word, character)
        else:
            target = self.target[index]
            return (input_ids, attention_mask, word, character, target)
        

class LLMModel(nn.Module):
    def __init__(self):
        super().__init__()

        model_config = AutoConfig.from_pretrained(config["transformer_model"]["base_transformer_model"])
        model_config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})                       

        self.roberta = AutoModel.from_pretrained(
            config["transformer_model"]["base_transformer_model"], config=model_config)

        self.regressor = nn.Linear(768, 1)
        

    def forward(self, input_ids, attention_mask, words, characters):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
        last_hidden = outputs.last_hidden_state   # [batch, seq_len, 768]
    
        cls_representation = last_hidden[:, 0, :]  # [batch, 768]
    
        logits = self.regressor(cls_representation)  # [batch, 1]

        # return probs.squeeze(-1)   # [batch]

        return logits
    

def predict(model, data_loader):
    """Returns an np.array with predictions of the |model| on |data_loader|"""
    model.eval()

    result = np.zeros(len(data_loader.dataset))    
    index = 0

    with torch.no_grad():
        for batch_num, (input_ids, attention_mask, word, character) in enumerate(data_loader):
            input_ids = input_ids.to(config["transformer_model"]["DEVICE"])
            attention_mask = attention_mask.to(config["transformer_model"]["DEVICE"])
            word = word.to(config["transformer_model"]["DEVICE"])
            character = character.to(config["transformer_model"]["DEVICE"])
                        
            pred = model(input_ids, attention_mask, word, character).flatten().to("cpu")

            result[index : index + pred.shape[0]] = pred
            index += pred.shape[0]

    return result


def load_router_model():

    global router_model

    if router_model is None:

        model = LLMModel().to(config["transformer_model"]["DEVICE"])

        model.load_state_dict(
            torch.load("data/models/model_ROBERTA_BASE.pth", map_location="cpu")
        )

        model.eval()

        router_model = model

    return router_model
