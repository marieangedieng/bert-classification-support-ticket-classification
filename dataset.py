import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenization encodant l'input text et l'attention mask
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long) # torch.long requis par CrossEntropyLoss
        }

def charger_datasets(csv_path="dataset/support-ticket-classification.csv", 
                     model_name="bert-base-uncased", 
                     max_length=256, 
                     test_size=0.2, 
                     seed=42):
    """Charge le CSV, prépare les features textuelles et sépare en Train/Val."""
    df = pd.read_csv(csv_path)
    
    # Concaténation obligatoire demandée : subject + text
    df['combined_text'] = df['subject'].fillna('') + " " + df['text'].fillna('')
    
    # Encodage dynamique des labels textuels en entiers
    df['label'] = df['label'].astype('category')
    classes = df['label'].cat.categories.tolist()
    label2id = {label: idx for idx, label in enumerate(classes)}
    id2label = {idx: label for idx, label in enumerate(classes)} # <-- Correction : idx (entier) en clé, label (texte) en valeur
    df['label_id'] = df['label'].cat.codes

    # Extraction des listes de données
    texts = df['combined_text'].tolist()
    labels = df['label_id'].tolist()

    # Split Train/Validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)

    return train_dataset, val_dataset, classes, label2id, id2label