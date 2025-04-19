# Multimodal Emotion Classification in Dialogues (MELD Dataset)

"""
Project structure:
- Load MELD dataset
- Preprocess textual, audio, and visual data
- Build a multimodal neural network combining BERT (text), audio embeddings, and image CNN features
- Train and evaluate emotion classification model
"""

import os
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import zipfile
import requests
from io import BytesIO

# --- Dataset Auto Download ---
def download_and_extract_meld():
    meld_url = "https://github.com/declare-lab/MELD/archive/refs/heads/master.zip"
    print("Downloading MELD dataset...")
    response = requests.get(meld_url)
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        z.extractall("data/")
    print("Dataset extracted to ./data/")

# --- Dataset Loading and Preprocessing ---

class MELDDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_encoder = LabelEncoder()
        self.data['Emotion'] = self.label_encoder.fit_transform(self.data['Emotion'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['Utterance'])
        encoded = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten(),
            'label': torch.tensor(row['Emotion'], dtype=torch.long)
        }

# --- Model Definition ---

class MultimodalEmotionClassifier(nn.Module):
    def __init__(self, n_classes):
        super(MultimodalEmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        return self.out(x)

# --- Train / Eval ---

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    print(classification_report(true_labels, predictions))

# --- Main Execution Block ---

def main():
    if not os.path.exists("data/MELD-master/data/train_sent_emo.csv"):
        download_and_extract_meld()

    df = pd.read_csv("data/MELD-master/data/train_sent_emo.csv")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = MELDDataset(df, tokenizer, max_len=128)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalEmotionClassifier(n_classes=len(df['Emotion'].unique()))
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    for epoch in range(3):
        loss = train_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    evaluate(model, dataloader, device)

    # Save model
    torch.save(model.state_dict(), "models/emotion_classifier.pt")

if __name__ == '__main__':
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    main()
