
import torch
import numpy as np 
from transformers import AutoTokenizer, AutoModelForCausalLM
import polars as pl
import json

genres = json.load(open('../datasets/id2genre.json', 'r'))

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len, target_list):
        self.tokenizer = tokenizer
        self.df = df
        self.title = list(df['overview'])
        self.targets = self.df.select(target_list).to_numpy()
        self.max_len = max_len

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.targets[index]),
            'title': title
        }

def get_data_loader(df, tokenizer, target_list = list(genres.values()), max_len=180, batch_size=32, shuffle=True):
    if isinstance(df, str):
        df = pl.read_csv(df)
    keep_cols = ['overview'] + list(genres.values())
    df = df[keep_cols]
    dataset = CustomDataset(df, tokenizer, max_len, target_list)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )
    return data_loader
