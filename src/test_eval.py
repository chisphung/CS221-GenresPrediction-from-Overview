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
    f1, jaccard, hamming = evaluate("./weights/distilled_bert.pt", test_df, target_list_path, base_model="distilbert-base-uncased")
    print(f"F1 Score: {f1}")
    print(f"Jaccard Score: {jaccard}")
    print(f"Hamming Loss: {hamming}")
    # Confusion Matrix
    y_true = test_df.select(pl.col("labels").to_list()).to_numpy().flatten()
    y_pred = test_df.select(pl.col("predictions").to_list()).to_numpy().flatten()
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(19)))
    disp.plot(cmap='Blues')
    disp.ax_.set_title("Confusion Matrix")
    disp.ax_.set_xlabel("Predicted Labels")
    disp.ax_.set_ylabel("True Labels")
    disp.figure_.set_size_inches(10, 10)
    disp.figure_.savefig("./confusion_matrix.png")
if __name__ == "__main__":
    main()