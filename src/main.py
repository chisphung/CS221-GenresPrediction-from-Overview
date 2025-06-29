from sklearn.model_selection import train_test_split, cross_val_score
from transformers import AutoTokenizer
import torch
from tools.load_model import load_model


model_path = r'C:\Users\User\Downloads\NLP\FinalTerm_Project\weights\bert_based_best.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer = load_model(model_path, device)

def single_predict(model, tokenizer, text, device, max_length=180, threshold=0.5):
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs.get('token_type_ids', None)  
        )
    
    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).long()

    return preds.cpu().numpy(), probs.cpu().numpy()


if __name__ == "__main__":
    text ="Earth's mightiest heroes must come together and learn to fight as a team if they are going to stop the mischievous Loki and his alien army from enslaving humanity."
    predictions, probabilities = single_predict(model, tokenizer, text, device)
    print("Predicted Labels:", predictions)
    print("Probabilities:", probabilities)