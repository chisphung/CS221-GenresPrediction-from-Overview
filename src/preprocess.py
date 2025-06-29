from datasets import load_dataset
import polars as pl
import re, string
import contractions
import emot
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

PUNCT_TO_REMOVE = str(string.punctuation + string.digits)
html_pattern = re.compile('<.*?>')
url_pattern = re.compile(r'https?://\S+|www\.\S+')
emot_obj = emot.emot()
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}

def remove_html(text):
    return html_pattern.sub(r' ', text)

def remove_urls(text):
    return url_pattern.sub(r' ', text)

def expand_contractions(text):
   expanded_text = contractions.fix(text)
   return expanded_text

def remove_punc_and_num(text):
    """custom function to remove the punctuation"""
    for token in PUNCT_TO_REMOVE:
        text = text.replace(token, "")
    return text

def remove_special_characters(text):
    text = re.sub('[^a-zA-z0-9\s]', ' ', text)
    return text

def handle_emoticons(text, remove_emoticon=True):
    dict_emoticons = dict(zip(emot_obj.emoticons(text)['value'], emot_obj.emoticons(text)['mean']))
    res_emoticons =  dict(sorted(dict_emoticons.items(), key = lambda kv:len(kv[1]), reverse=True))
    for emoticon, mean in res_emoticons.items():
        if remove_emoticon:
            text = text.replace(emoticon, " ")
        else:
            text = text.replace(emoticon, mean)
    return text

def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.VERB)) for word, pos in pos_tagged_text])

def preprocess_text(text):
    text = remove_html(text)
    text = remove_urls(text)
    text = expand_contractions(text)
    text = text.lower()
    text = remove_punc_and_num(text)
    text = remove_special_characters(text)
    text = handle_emoticons(text)
    text = remove_stopwords(text)
    text = lemmatize_words(text)
    return text

if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset("csv", data_files="data/train.csv")

    df = pl.DataFrame(dataset["train"])
    df = df.drop_nulls()
    df = df.select(df.columns[:5])
    df = df.filter(~(pl.col('genres').is_null() | pl.col('overview').is_null()))

    # One hot encoding
    distinct_genres = set()
    for genre in df['genres']:
        genre = str.split(genre, '-')
        distinct_genres.update(genre)
    
    for genre in distinct_genres:
        df = df.with_columns(
            pl.when(pl.col("genres").str.contains(genre))
            .then(1)
            .otherwise(0)
            .alias(genre)
        )
    
    genre_columns = [col for col in df.columns if col not in ['overview', 'id', 'title', 'original_language', 'genres']] # Filter out non-genre columns
    df = df.select(['overview'] + genre_columns)

    # Preprocess the overview text
    df_preprocessed = df.with_columns(
    pl.col('overview').map_elements(preprocess_text).alias('overview')
    )
    df_preprocessed.write_csv("datasets/preprocessed_dataset.csv")
    # Trim the dataset

    max_per_genre = 45000

    genre_cols = df_preprocessed.columns[1:]  # Remove 'overview' col
    genre_counter = {genre: 0 for genre in genre_cols}

    kept_rows = []

    for row in df_preprocessed.iter_rows(named=True):
        genres = [genre for genre in genre_cols if row[genre] == 1]

        if all(genre_counter[genre] < max_per_genre for genre in genres):
            kept_rows.append(row)
            for genre in genres:
                genre_counter[genre] += 1

    df_trimmed = pl.DataFrame(kept_rows)
    # Save the preprocessed and trimmed dataset
    df_trimmed.write_csv("datasets/preprocessed_trimmed.csv")