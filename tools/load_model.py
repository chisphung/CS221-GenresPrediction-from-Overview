import torch
from transformers import AutoTokenizer, BertModel

class BERTClass(torch.nn.Module):
    def __init__(self, pretrain_model_name='bert-base-uncased', num_labels=19):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained(pretrain_model_name, return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        return self.linear(output_dropout)  # return raw logits

def load_model(model_path, device = 'cuda' if torch.cuda.is_available() else 'cpu', num_labels=19, base_model='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = BERTClass(pretrain_model_name=base_model, num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer
