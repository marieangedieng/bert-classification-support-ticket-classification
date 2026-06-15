import random
import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed=42):
    """fixe les graines aléatoires pour assurer la reproductibilité"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(labels, preds):
    """calcule accuracy et le F1-Score """
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return acc, f1

def plot_confusion_matrix(labels, preds, classes, title="Matrice de Confusion"):
    """genere et renvoie la matrice de confusion"""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax)
    plt.ylabel('Vrais Labels')
    plt.xlabel('Prédictions')
    plt.title(title)
    plt.tight_layout()
    return fig