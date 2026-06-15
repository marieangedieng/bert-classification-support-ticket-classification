import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import json

from dataset import charger_datasets
from model import get_bert_model
from utils import set_seed, compute_metrics, plot_confusion_matrix

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Entrainement d'une époque du modèle BERT"""
    model.train()

    total_loss=0
    all_preds=[]
    all_labels=[]

    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()

        # Gradient clipping pour stabiliser l'entraînement de BERT
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss  += loss.item()
        preds = torch.argmax(logits, dim=1).cpu.numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = total_loss / len(dataloader)
    acc, f1 = compute_metrics(all_labels, all_preds)
    return epoch_loss, acc, f1

def eval_epoch(model, dataloader, device):
    """Evaluation d'une époque du modèle BERT"""
    model.eval()

    total_loss=0
    all_preds=[]
    all_labels=[]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss+=loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = total_loss / len(dataloader)
    acc, f1 = compute_metrics(all_labels, all_preds)
    return epoch_loss, acc, f1, all_preds, all_labels

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device de calcul : {device}")

    CONFIG= {
        "model_name": "bert-base-uncased",
        "epochs": 4,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "max_length": 256,
        "seed": 42
    }

    # Préparation des datasets
    train_dataset, val_dataset, classes, label2id, id2label = charger_datasets(
        csv_path="dataset/support-ticket-classification.csv",
        model_name=CONFIG["model_name"],
        max_length=CONFIG["max_length"],
        seed=CONFIG["seed"]
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
            
    # Initialisation du modèle
    model = get_bert_model(
        model_name=CONFIG["model_name"], 
        num_labels=len(classes), 
        label2id=label2id, 
        id2label=id2label
    ).to(device)

    # Configuration de l'optimiseur et du scheduler d'apprentissage
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    total_steps = len(train_loader) * CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Sauvegarde locale des dictionnaires de correspondance pour la démo Gradio
    with open("label_mapping.json", "w") as f:
        json.dump({"classes": classes, "label2id": label2id, "id2label": id2label, "model_name": CONFIG["model_name"]}, f)

    # Suivi centralisé avec WandB
    run = wandb.init(project="Devoir_3_BERT", name="BERT_Ticket_Classification", config=CONFIG)

    best_val_loss = float('inf')
    best_preds, best_labels = [], []

    for epoch in range(CONFIG["epochs"]):
        print(f"\n--- Époque {epoch + 1}/{CONFIG['epochs']} ---")
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, val_f1, preds, labels = eval_epoch(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_accuracy": train_acc, "train_f1": train_f1,
            "val_loss": val_loss, "val_accuracy": val_acc, "val_f1": val_f1
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_preds, best_labels = preds, labels

            # Sauvegarde complète du dossier compatible HuggingFace
            model.save_pretrained("best_bert_model")
            print("Nouveau meilleur checkpoint sauvegardé dans 'best_bert_model' !")

    # Log de la matrice de confusion finale sur WandB
    fig_cm = plot_confusion_matrix(best_labels, best_preds, classes=classes, title="Matrice de Confusion Finale - BERT")
    wandb.log({"confusion_matrix": wandb.Image(fig_cm)})
    run.finish()

if __name__ == "__main__":
    main()
        