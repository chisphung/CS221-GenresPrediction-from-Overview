from predict import get_predictions
from data_loader import get_data_loader
from load_model import load_model
from postprocess import post_process
from sklearn.metrics import f1_score, jaccard_score, hamming_loss, classification_report
import json
import torch
import polars as pl

genres = json.load(open('../datasets/id2genre.json', 'r'))

def evaluate(model_path, data_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Evaluate the model on the given data loader.
    """
    if isinstance(data_path, str):
        df = pl.read_csv(data_path)
    model, tokenizer = load_model(model_path, device=device)
    data_loader = get_data_loader(df, tokenizer, target_list=list(genres.values()), max_len=180, batch_size=32, shuffle=False)
    all_predictions = []
    all_targets = []

    titles, predictions, prediction_probs, target_values = get_predictions(model, data_loader, device=device)
    test_labels = df.select(list(genres.values())).to_numpy()
    print(classification_report(test_labels, predictions, target_names=list(genres.values())))

    f1_score_value = f1_score(test_labels, predictions, average='micro')
    jaccard_score_value = jaccard_score(test_labels, predictions, average='samples')
    hamming_loss_value = hamming_loss(test_labels, predictions)

    return f1_score_value, jaccard_score_value, hamming_loss_value

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python eval.py <model_path> <data_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    data_path = sys.argv[2]

    f1, jaccard, hamming = evaluate(model_path, data_path)
    print(f"F1 Score: {f1}")
    print(f"Jaccard Score: {jaccard}")
    print(f"Hamming Loss: {hamming}")