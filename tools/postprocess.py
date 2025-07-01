import torch
import numpy as np 
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

def post_process(probs, k=3, use_threshold=True):
    probs = probs.cpu().numpy()

    if use_threshold:
        class_thresholds = np.array([
            0.5,  # 0. Comedy
            0.3,  # 1. Fantasy
            0.3,  # 2. Crime
            0.3,  # 3. Family
            0.5,  # 4. Horror
            0.3,  # 5. TV Movie
            0.3,  # 6. Action
            0.6,  # 7. Animation
            0.3,  # 8. War
            0.6,  # 9. Documentary
            0.6,  # 10. Western
            0.3,  # 11. History
            0.3,  # 12. Thriller
            0.3,  # 13. Mystery
            0.6,  # 14. Music
            0.3,  # 15. Romance
            0.5,  # 16. Drama
            0.3,  # 17. Adventure
            0.5   # 18. Science Fiction 
        ])


        preds = (probs > class_thresholds).astype(int)

    # Top-k logic
    topk_indices = np.argsort(probs[0])[::-1][:k]
    topk_preds = np.zeros_like(preds)
    topk_preds[0, topk_indices] = 1

    final_preds = np.logical_or(preds, topk_preds).astype(int)

    # Custom rule for class 8
    if probs[0, 8] < 0.7:
        final_preds[0, 8] = 0

    return final_preds
