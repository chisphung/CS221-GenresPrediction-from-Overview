from tools.eval import evaluate
from sklearn.model_selection import train_test_split
import polars as pl 

def main():
    df = pl.read_csv("./datasets/train.csv")
    train_df, test_df = train_test_split(df, test_size=0.3)
    val_df, test_df = train_test_split(test_df, test_size=0.5)
    f1, jaccard, hamming = evaluate("./weights/best_model.pt", test_df)
    print(f"F1 Score: {f1}")
    print(f"Jaccard Score: {jaccard}")
    print(f"Hamming Loss: {hamming}")

if __name__ == "__main__":
    main()