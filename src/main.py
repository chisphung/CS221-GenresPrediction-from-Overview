from sklearn.model_selection import train_test_split, cross_val_score
from transformers import AutoTokenizer
import torch
from tools.load_model import load_model
from tools.postprocess import post_process
import json
import numpy as np
from pathlib import Path

def single_predict(model, tokenizer, text, device, max_length=180, threshold=0.5):
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs.get('token_type_ids', None)  
        )
    
    probs = torch.sigmoid(outputs)
    preds = post_process(probs, k=3)

    return preds, probs


if __name__ == "__main__":
    path = Path("datasets") / "id2genre_bert_cased.json"
    with open(path, "r") as f:
        id2genre = json.load(f)

    model_path = r'./weights/bert_cased.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(model_path, device, num_labels=len(id2genre), base_model='bert-base-cased')
    text ="Earth's mightiest heroes must come together and learn to fight as a team if they are going to stop the mischievous Loki and his alien army from enslaving humanity."
    predictions, probabilities = single_predict(model, tokenizer, text, device)
    pred_indices = np.where(predictions[0] == 1)[0]
    print("Predicted Labels:", [id2genre[str(i)] for i in pred_indices])
    print("Probabilities:", probabilities[0][pred_indices].tolist())