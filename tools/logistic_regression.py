import polars as pl
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":

    df = pl.read_csv("datasets/train.csv")
    df = df.filter(~(pl.col('genres').is_null() | pl.col('overview').is_null()))
    df = df.select(['overview', 'genres'])

    df = df.copy()
    labels = df.iloc[:, 1:-1].values
    df['labels'] = list(labels)

    corpus = list(df['overview'].dropna())
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(corpus).toarray()

    df['overview'] = list(vectors)
    df = df.loc[:, ['overview','labels']]

    TEST_SIZE = 0.2
    split_idx = int(TEST_SIZE * len(vectors))
    Trainvecs = vectors[:split_idx]
    Testvecs = vectors[split_idx:]
    Trainclss = labels[:split_idx]
    Testclss = labels[split_idx:]

    cls = MultiOutputClassifier(LogisticRegression())
    cls.fit(Trainvecs, Trainclss)

    joblib.dump(cls, './weights/logistic_regression_model.joblib')