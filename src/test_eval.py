from tools.eval import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import polars as pl 
import json

def main():
    df = pl.read_csv("./datasets/train.csv")
    train_df, test_df = train_test_split(df, test_size=0.3)
    val_df, test_df = train_test_split(test_df, test_size=2/3)
    target_list_path = "./datasets/id2genre_distillBERT.json"
    # target_list_path = "./datasets/id2genre_bert_cased.json"
    # target_list_path = "./datasets/id2genre_bert_uncased.json"
    genres = json.load(open(target_list_path, 'r'))
    f1, jaccard, hamming = evaluate("./weights/distilled_bert.pt", test_df, target_list_path, base_model="distilbert-base-uncased")
    print(f"F1 Score: {f1}")
    print(f"Jaccard Score: {jaccard}")
    print(f"Hamming Loss: {hamming}")

if __name__ == "__main__":
    main()