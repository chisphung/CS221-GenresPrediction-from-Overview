import sys
import os
import torch
import streamlit as st

# Ensure root path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.main import single_predict
from tools.load_model import load_model
from tools.postprocess import post_process

# Mapping from label index to genre
ID_TO_GENRE = {
    0: 'Action', 1: 'Adventure', 2: 'Animation', 3: 'Comedy', 4: 'Crime',
    5: 'Documentary', 6: 'Drama', 7: 'Family', 8: 'Fantasy', 9: 'History',
    10: 'Horror', 11: 'Music', 12: 'Mystery', 13: 'Romance', 14: 'Science Fiction',
    15: 'TV Movie', 16: 'Thriller', 17: 'War', 18: 'Western'
}

def main():
    st.title("🎬 Movie Genre Prediction from Overview")
    
    text = st.text_area("Enter the movie overview:")
    model_selection = st.selectbox("Choose a base model:", [
        "bert-base-uncased",
        "distilled-bert-base-uncased",
        "roberta-base"
    ])
    if model_selection == "bert-base-uncased":
        model_path = "weights/bert_based.pt"
    elif model_selection == "distilled-bert-base-uncased":
        model_path = "weights/disstilledBERT.pt"
    elif model_selection == "roberta-base":
        model_path = "weights/roberta_base.pt"

    if st.button("Predict Genres"):
        if not text.strip():
            st.warning("Please enter some text to classify.")
            return
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer = load_model(model_path, device)

        preds, probs = single_predict(model, tokenizer, text, device)
        top_preds = post_process(probs, k=3)
        
        st.success("Prediction Complete")
        st.markdown("### Top Predicted Genres:")
        for i, pred in enumerate(top_preds[0]):
            if pred == 1:
                genre = ID_TO_GENRE[i]
                prob = probs[0][i].item()
                st.write(f"{genre}: {prob:.2f}")


if __name__ == "__main__":
    main()
