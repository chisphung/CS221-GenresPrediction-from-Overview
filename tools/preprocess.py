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

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r' ', text)

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r' ', text)

def expand_contractions(text):
   expanded_text = contractions.fix(text)
   return expanded_text

def remove_punc_and_num(text):
    """custom function to remove the punctuation"""
    PUNCT_TO_REMOVE = str(string.punctuation + string.digits)
    for token in PUNCT_TO_REMOVE:
        text = text.replace(token, "")
    return text

def remove_special_characters(text):
    text = re.sub('[^a-zA-z0-9\s]', ' ', text)
    return text

def handle_emoticons(text, remove_emoticon=True):
    emot_obj = emot.emot()
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
    STOPWORDS = set(stopwords.words('english'))
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


def lemmatize_words(text):
    wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
    lemmatizer = WordNetLemmatizer()
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

def trim_dataset(df_train, max_per_genre=20000, preserve_labels = []):
    genre_cols = df_train.columns[1:]
    genre_counter = {genre: 0 for genre in genre_cols}
        
    df_shuffled = df_train.sample(fraction=1.0, shuffle=True, seed=42)
    kept_rows = []
    
    for row in df_shuffled.iter_rows(named=True):
        genres = [genre for genre in genre_cols if row[genre] == 1]
        if any(label in preserve_labels for label in genres):
            kept_rows.append(row)
            for genre in genres:
                genre_counter[genre] += 1
    
    for row in df_shuffled.iter_rows(named=True):
        genres = [genre for genre in genre_cols if row[genre] == 1]
        if row in kept_rows:
            continue
         
        if any(label in preserve_labels for label in genres):
            continue
         
        if all(genre_counter[genre] < max_per_genre for genre in genres):
            kept_rows.append(row)
            for genre in genres:
                genre_counter[genre] += 1
    
    df_trimmed = pl.DataFrame(kept_rows)
    return df_trimmed

if __name__ == "__main__":
    # Load the dataset
    # dataset1 = load_dataset('wykonos/movies')

    df = pl.read_csv('hf://datasets/wykonos/movies/movies_dataset.csv')
    print(len(df['genres']))
    # df = df.drop_nulls() # Cannot use drop nulls for all columns because there are some unused columns with null values
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
    pl.col('overview').map_elements(preprocess_text, return_dtype=str).alias('overview')
    )
    # print(len(df_preprocessed['overview']))
    df_preprocessed.write_csv("datasets/preprocessed_dataset.csv")
    # Trim the dataset
    df_trimmed = trim_dataset(df_preprocessed, max_per_genre = 20000, preserve_labels=['War', 'History', 'Western'])
    df_trimmed.write_csv("datasets/trimmed_dataset.csv")

