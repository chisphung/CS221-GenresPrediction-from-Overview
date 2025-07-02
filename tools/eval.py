from tools.predict import get_predictions
from tools.data_loader import get_data_loader
from tools.load_model import load_model
from tools.postprocess import post_process
from tools.ensemble import ensemble_predict
from sklearn.metrics import f1_score, jaccard_score, hamming_loss, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
import torch
import polars as pl


def evaluate(model_path, df, target_list_path, base_model = "bert-base-uncased", device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Evaluate the model on the given data loader.
    """
    print(f"Using device: {device}")
    genres = json.load(open(target_list_path, 'r'))
    if isinstance(df, str):
        df = pl.read_csv(df)
    model, tokenizer = load_model(model_path, base_model=base_model, device=device)
    data_loader = get_data_loader(df, tokenizer, target_list=list(genres.values()), max_len=180, batch_size=32, shuffle=False)
    all_predictions = []
    all_targets = []

    titles, predictions, prediction_probs, target_values = get_predictions(model, data_loader, target_list_path=target_list_path, device=device)
    # predictions = post_process(predictions)
    predictions = post_process(prediction_probs)
    print("Post-processing completed.")
    test_labels = df.select(list(genres.values())).to_numpy()
    print(classification_report(test_labels, predictions, target_names=list(genres.values())))

    f1_score_value = f1_score(test_labels, predictions, average='micro')
    jaccard_score_value = jaccard_score(test_labels, predictions, average='samples')
    hamming_loss_value = hamming_loss(test_labels, predictions)
    # Save predictions results
    json.dump({
        "titles": titles,
        "predictions": predictions.tolist(),
        "prediction_probs": prediction_probs.tolist(),
        "targets": target_values.tolist()
    }, open("predictions.json", "w"), indent=4)

    return f1_score_value, jaccard_score_value, hamming_loss_value

def eval_ennsemble(df, target_list_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Evaluate the ensemble model on the given data loader.
    """
    print(f"Using device: {device}")
    genres = json.load(open(target_list_path, 'r'))
    if isinstance(df, str):
        df = pl.read_csv(df)
    predictions = []
    for sample in df['overview']:
        _, pred = ensemble_predict(str(sample), device=device)
        predictions.append(pred)
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