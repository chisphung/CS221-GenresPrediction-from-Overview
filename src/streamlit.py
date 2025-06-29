import streamlit as st
from main import single_predict
from tools.load_model import load_model
import torch
from tools.postprocess import post_process

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
id_to_genre = {
    0: 'Action',
    1: 'Adventure',
    2: 'Animation',
    3: 'Comedy',
    4: 'Crime',
    5: 'Documentary',
    6: 'Drama',
    7: 'Family',
    8: 'Fantasy',
    9: 'History',
    10: 'Horror',
    11: 'Music',
    12: 'Mystery',
    13: 'Romance',
    14: 'Science Fiction',
    15: 'TV Movie',
    16: 'Thriller',
    17: 'War',
    18: 'Western'
}


def main():
    st.title("Text Classification with BERT")
    text = st.text_area("Enter text for classification:")
    model_selection = st.selectbox("Select Model",
        ["bert-base-uncased", "distilled-bert-base-uncased", "roberta-base"]
    )
    model, tokenizer = load_model(
        model_path=f"weights/{model_selection}_best.pt",
        device="cuda" if torch.cuda.is_available() else "cpu",
        base_model=model_selection
    )
    if st.button("Classify"):
        preds, probs = single_predict(model, tokenizer, text, device)
        preds = post_process(probs, k=3)
        st.write("Predicted Labels:", [id_to_genre[i] for i in preds])
        st.write("Probabilities:", probs)