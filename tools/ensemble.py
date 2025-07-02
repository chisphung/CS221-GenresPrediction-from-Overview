from tools.load_model import load_model
from tools.postprocess import post_process
from src.main import single_predict
import torch
import json
import numpy as np

def ensemble_predict(text, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Load label maps
    cased_genres = json.load(open("datasets/id2genre_bert_cased.json"))
    uncased_genres = json.load(open("datasets/id2genre_bert_uncased.json"))
    distilled_genres = json.load(open("datasets/id2genre_distillBERT.json"))  # This will be our global decoder

    # Load models
    bert_cased_model, bert_cased_tokenizer = load_model("weights/bert_cased.pt", device, num_labels=len(cased_genres), base_model='bert-base-cased')
    bert_uncased_model, bert_uncased_tokenizer = load_model("weights/bert_uncased.pt", device, num_labels=len(uncased_genres), base_model='bert-base-uncased')
    distilled_bert_model, distilled_bert_tokenizer = load_model("weights/distilled_bert.pt", device, num_labels=len(distilled_genres), base_model='distilbert-base-uncased')

    # Predictions
    cased_pred, _ = single_predict(bert_cased_model, bert_cased_tokenizer, text, device)
    uncased_pred, _ = single_predict(bert_uncased_model, bert_uncased_tokenizer, text, device)
    distilled_pred, _ = single_predict(distilled_bert_model, distilled_bert_tokenizer, text, device)

    # Decode to label strings
    decoded_cased = {cased_genres[str(i)] for i in range(len(cased_genres)) if cased_pred[0][i] == 1}
    decoded_uncased = {uncased_genres[str(i)] for i in range(len(uncased_genres)) if uncased_pred[0][i] == 1}
    decoded_distilled = {distilled_genres[str(i)] for i in range(len(distilled_genres)) if distilled_pred[0][i] == 1}

    # Voting per label
    vote_counter = {}
    for genre in set(distilled_genres.values()):
        vote_counter[genre] = 0
        if genre in decoded_cased:
            vote_counter[genre] += 1
        if genre in decoded_uncased:
            vote_counter[genre] += 1
        if genre in decoded_distilled:
            vote_counter[genre] += 1

    # Convert to binary array aligned to distilled_genres (sorted by id)
    output_array = np.zeros(len(distilled_genres), dtype=int)
    for idx, genre in distilled_genres.items():
        if vote_counter.get(genre, 0) >= 2:
            output_array[int(idx)] = 1  # Ensure index is int

    # Human-readable summary
    decoded_labels = [genre for genre, count in vote_counter.items() if count >= 2]
    print_string = "Predicted Labels: " + ", ".join(decoded_labels)

    return print_string, output_array

if __name__ == "__main__":
    text = "Earth's mightiest heroes must come together and learn to fight as a team if they are going to stop the mischievous Loki and his alien army from enslaving humanity."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = ensemble_predict(text, device)
    print("Ensemble Predictions:", predictions)
    # pred_indices = torch.where(predictions[0] == 1)[0]