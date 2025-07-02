import polars as pl
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, jaccard_score, hamming_loss

if __name__ == "__main__":
    df = pl.read_csv("datasets/train.csv")
    df = df.filter(~pl.col("overview").is_null())
    overview = df["overview"].to_list()

    # Label columns are all columns from 'Comedy' 
    label_cols = df.columns[df.columns.index("Comedy"):]
    labels = df.select(label_cols).to_numpy()

    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(overview).toarray()
    TEST_SIZE = 0.2
    split_idx = int(TEST_SIZE * len(vectors))
    Trainvecs = vectors[:split_idx]
    Testvecs = vectors[split_idx:]
    Trainclss = labels[:split_idx]
    Testclss = labels[split_idx:]

    model = joblib.load("./weights/log_final.joblib")
    pred_test = model.predict(Testvecs)
    print(classification_report(Testclss, pred_test, target_names=label_cols))
    print("F1 Score:", f1_score(Testclss, pred_test, average='micro'))
    print("Jaccard Score:", jaccard_score(Testclss, pred_test, average='samples'))
    print("Hamming Loss:", hamming_loss(Testclss, pred_test))

