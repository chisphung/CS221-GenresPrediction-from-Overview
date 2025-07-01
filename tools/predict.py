
import torch
import numpy as np 
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
from load_model import load_model
from data_loader import get_data_loader
import pandas as pd
import json 

genres = json.load(open('../datasets/id2genre.json', 'r'))

def get_predictions(model, data_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Outputs:
      predictions - 
    """
    model = model.eval()
    
    titles = []
    predictions = []
    prediction_probs = []
    target_values = []

    with torch.no_grad():
      for data in data_loader:
        title = data["title"]
        ids = data["input_ids"].to(device, dtype = torch.long)
        mask = data["attention_mask"].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data["targets"].to(device, dtype = torch.float)
        
        outputs = model(ids, mask, token_type_ids)
        # add sigmoid, for the training sigmoid is in BCEWithLogitsLoss
        outputs = torch.sigmoid(outputs).detach().cpu()
        # thresholding at 0.5
        preds = outputs.round()
        targets = targets.detach().cpu()

        titles.extend(title)
        predictions.extend(preds)
        prediction_probs.extend(outputs)
        target_values.extend(targets)
    
    predictions = torch.stack(predictions)
    prediction_probs = torch.stack(prediction_probs)
    target_values = torch.stack(target_values)
    
    return titles, predictions, prediction_probs, target_values

if __name__ == "__main__":
  if len(sys.argv) < 3:
       print("Usage: python predict.py <model_path> <data>")
       sys.exit(1)

  model_path = sys.argv[1]
  data = sys.argv[2]

  model, tokenizer = load_model(model_path)
  data_loader = get_data_loader(data, tokenizer)

  titles, predictions, prediction_probs, target_values = get_predictions(model, data_loader)
  print("Predictions completed.")
  print(f"Predictions: {predictions}")
  for index, pred in enumerate(predictions):
     print(f"Sample {index + 1}:")
     result = []
     for genre_index, genre in enumerate(pred):
        if genre == 1:
           result.append(genres[str(genre_index)])
     print(result)
           
        

   # Process and save the predictions as needed