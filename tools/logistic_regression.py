import polars as pl
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df = pl.read_csv("datasets/train.csv")
    df = df.filter(~pl.col("overview").is_null())
    overview = df["overview"].to_list()

    # Label columns are all columns from 'Comedy' 
    label_cols = df.columns[df.columns.index("Comedy"):]
    labels = df.select(label_cols).to_numpy()

    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(overview).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)


    joblib.dump(model, "./weights/logistic_regression_model.joblib")
    joblib.dump(tfidf, "./weights/tfidf_vectorizer.joblib")
