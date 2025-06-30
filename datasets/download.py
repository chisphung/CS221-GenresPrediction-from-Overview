import gdown
import os

if __name__ == "__main__":
    output_path = "datasets/preprocessed_dataset.csv"
    gdown.download("https://drive.google.com/uc?id=1-9XtB-Nn_1xinwpRug68O-WLFuSThhOo", output_path, quiet=False)
    print("Dataset downloaded successfully to:", output_path)
