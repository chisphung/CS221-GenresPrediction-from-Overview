import torch
import numpy as np 
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

def post_process(probs, k=3, use_threshold=True):
    probs = probs.numpy()

    if use_threshold:
        class_thresholds = np.array([
            0.5,  # class 0
            0.3,  # class 1
            0.5,  # class 2
            0.5,  # class 3
            0.5,  # class 4
            0.5,  # class 5
            0.3,  # class 6
            0.4,  # class 7
            0.3,  # class 8
            0.5,  # class 9
            0.3,  # class 10
            0.4,  # class 11
            0.5,  # class 12
            0.3,  # class 13
            0.5,  # class 14
            0.5,  # class 15
            0.4,  # class 16
            0.5,  # class 17 
            0.5   # class 18 
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
