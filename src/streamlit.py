import streamlit as st
from main import single_predict
from tools.load_model import load_model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        st.write("Predicted Labels:", preds)
        st.write("Probabilities:", probs)