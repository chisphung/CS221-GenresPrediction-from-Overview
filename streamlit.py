import sys
import os
import torch
import streamlit as st

# Ensure root path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.main import single_predict
from tools.load_model import load_model
from tools.postprocess import post_process
from tools.ensemble import ensemble_predict
import json

# Mapping from label index to genre
genres_bert_cased = json.load(open("datasets/id2genre_bert_cased.json"))
genres_bert_uncased = json.load(open("datasets/id2genre_bert_uncased.json"))
genres_distilled_bert = json.load(open("datasets/id2genre_distillBERT.json"))
def main():
    st.title("🎬 Movie Genre Prediction from Overview")
    
    text = st.text_area("Enter the movie overview:")
    model_selection = st.selectbox("Choose a base model:", [
        "bert-base-uncased",
        "distilbert-base-uncased",
        "bert-base-cased",
        "ensemble"
    ])
    if model_selection == "bert-base-uncased":
        ID_TO_GENRE = genres_bert_uncased
        model_path = "weights/bert_uncased.pt"
    elif model_selection == "distilbert-base-uncased":
        ID_TO_GENRE = genres_distilled_bert
        model_path = "weights/distilled_bert.pt"
    elif model_selection == "bert-base-cased":
        ID_TO_GENRE = genres_bert_cased
        model_path = "weights/bert_cased.pt"

    if st.button("Predict Genres"):
        if not text.strip():
            st.warning("Please enter some text to classify.")
            return
        if model_selection == "ensemble":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            preds = ensemble_predict(text, device)
            st.success("Prediction Complete")
            st.markdown("### Predicted Genres:")
            st.write(preds)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, tokenizer = load_model(model_path, device, base_model=model_selection)

            preds, probs = single_predict(model, tokenizer, text, device)
            top_preds = post_process(probs, k=3)
            
            st.success("Prediction Complete")
            st.markdown("### Top Predicted Genres:")
            for i, pred in enumerate(top_preds[0]):
                if pred == 1:
                    genre = ID_TO_GENRE[str(i)]
                    prob = probs[0][i].item()
                    st.write(f"{genre}: {prob:.2f}")


if __name__ == "__main__":
    main()
