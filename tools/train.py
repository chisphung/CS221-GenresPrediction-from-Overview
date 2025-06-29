import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset
import numpy as np 
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import BertModel
from sklearn.model_selection import train_test_split
# import wandb
# from get_token import wandb_token
from evaluate import load
import polars as pl
import sys, os
import tqdm.auto as tq
from collections import defaultdict

# wandb.login(key=wandb_token())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 180
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-05
THRESHOLD = 0.5 # threshold for the sigmoid

if len(sys.argv) < 3:
        print("Usage: python train.py <pretrained_model_name_or_path> <dataset_path>")
        sys.exit(1)
pretrain_model_name = sys.argv[1]
dataset_path = sys.argv[2]
max_length = 180
num_labels = 19
df = pl.read_csv(dataset_path)
df = df.drop("original_language", "id", "genres", "title")
label_cols = df.columns[-19:]

train_df, test_df = train_test_split(df, test_size = 0.3)
val_df, test_df = train_test_split(test_df, test_size = 2/3)

# train_df = train_df.to_numpy()
# test_df  = test_df.to_numpy()
# val_df = val_df.to_numpy()

target_list = list(df.columns)
target_list = target_list[1:]


def compute_pos_weight(df_labels: pl.DataFrame) -> torch.Tensor:
    label_counts = df_labels.select(pl.all().sum()).row(0)
    total_samples = df_labels.height
    pos_weight_list = [(total_samples - count) / count for count in label_counts]
    pos_weight = torch.tensor(pos_weight_list, dtype=torch.float)
    return pos_weight


def loss_fn(outputs, targets, pos_weight=None):
    return torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight.to(device))(outputs, targets)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len, target_list):
        self.tokenizer = tokenizer
        self.df = df
        self.title = list(df['overview'])
        self.targets = self.df.select(target_list).to_numpy()
        self.max_len = max_len

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.targets[index]),
            'title': title
        }
    
def train_model(training_loader, model, optimizer):

    losses = []
    correct_predictions = 0
    num_samples = 0
    # set model to training mode (activate droput, batch norm)
    model.train()
    # initialize the progress bar
    loop = tq.tqdm(enumerate(training_loader), total=len(training_loader), 
                      leave=True, colour='steelblue')
    for batch_idx, data in loop:
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        # forward
        outputs = model(ids, mask, token_type_ids) # (batch,predict)=(32,8)
        loss = loss_fn(outputs, targets, pos_weight=None)
        losses.append(loss.item())
        # training accuracy, apply sigmoid, round (apply thresh 0.5)
        outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
        targets = targets.cpu().detach().numpy()
        correct_predictions += np.sum(outputs==targets)
        num_samples += targets.size   # total number of elements in the 2D array

        # backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # grad descent step
        optimizer.step()

        # Update progress bar
        #loop.set_description(f"")
        #loop.set_postfix(batch_loss=loss)

    # returning: trained model, model accuracy, mean loss
    return model, float(correct_predictions)/num_samples, np.mean(losses)

def eval_model(validation_loader, model, optimizer):
    losses = []
    correct_predictions = 0
    num_samples = 0
    subset_correct = 0
    # set model to eval mode (turn off dropout, fix batch norm)
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader, 0):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)
            losses.append(loss.item())

            # validation accuracy
            # add sigmoid, for the training sigmoid is in BCEWithLogitsLoss
            outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
            targets = targets.cpu().detach().numpy()
            correct_predictions += np.sum(outputs==targets)
            num_samples += targets.size   # total number of elements in the 2D array

                # Subset accuracy (all labels correct per sample)
            subset_correct += np.sum(np.all(outputs == targets, axis=1))
        subset_acc = float(subset_correct) / num_samples

    return float(correct_predictions)/num_samples, np.mean(losses), subset_acc

class BERTClass(torch.nn.Module):
        def __init__(self):
            super(BERTClass, self).__init__()
            self.bert_model = BertModel.from_pretrained(pretrain_model_name, return_dict=True)
            self.dropout = torch.nn.Dropout(0.3)
            self.linear = torch.nn.Linear(768, 19)

        def forward(self, input_ids, attn_mask, token_type_ids):
            output = self.bert_model(
                input_ids, 
                attention_mask=attn_mask, 
                token_type_ids=token_type_ids
            )
            output_dropout = self.dropout(output.pooler_output)
            output = self.linear(output_dropout)
            return output


if __name__ == "__main__":
    pretrain_model_name = sys.argv[1]
    pos_weight = compute_pos_weight(df.select(label_cols))
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_name)

    train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN, target_list)
    val_dataset = CustomDataset(val_df, tokenizer, MAX_LEN, target_list)
    test_dataset = CustomDataset(test_df, tokenizer, MAX_LEN, target_list)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, 
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=0
    )

    val_data_loader = torch.utils.data.DataLoader(val_dataset, 
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_data_loader = torch.utils.data.DataLoader(test_dataset, 
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
        
    training_args = TrainingArguments(
        output_dir="out_dir",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        eval_steps=100,
        save_steps=100,
        logging_steps=100,
        save_total_limit=2,
        report_to="wandb", 
        logging_dir="./logs",
        eval_strategy="steps",  # or "epoch"
        # eval_steps=500,  # << adjust based on your dataset size
        load_best_model_at_end=True,  # really useful with early stopping
        metric_for_best_model="f1_macro",  # or "f1", "loss", etc. depending on your metric
        greater_is_better=True,  # set to False if you're minimizing (like loss)
    )
    
    model = BERTClass()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(1, EPOCHS+1):
        print(f'Epoch {epoch}/{EPOCHS}')
        model, train_acc, train_loss = train_model(train_data_loader, model, optimizer)
        val_acc, val_loss, val_subset_acc = eval_model(val_data_loader, model, optimizer)

        print(f'train_loss={train_loss:.4f}, val_loss={val_loss:.4f} train_acc={train_acc:.4f}, val_acc={val_acc:.4f}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        # save the best model
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), os.path.join("/weights/best_model.pt"))
            best_accuracy = val_acc
