import pandas as pd
import numpy as np
from transformers import AutoTokenizer

def inspecter_dataset(csv_path="dataset/support-ticket-classification.csv", model_name="bert-base-uncased"):
    print("ANALYSE EXPLORATOIRE DU DATASET")
    
    # 1. Chargement des données
    df = pd.read_csv(csv_path)
    
    # Recréer le texte combiné comme dans votre pipeline d'entraînement
    df['combined_text'] = df['subject'].fillna('') + " " + df['text'].fillna('')
    
    # 2. Nombre total d'exemples et de classes
    total_exemples = len(df)
    classes = df['label'].unique()
    nb_classes = len(classes)
    
    print(f"• Nombre total d'exemples : {total_exemples}")
    print(f"• Nombre de classes       : {nb_classes}")
    print(f"• Liste des classes       : {list(classes)}\n")
    
    # 3. Répartition des classes en pourcentage
    print("Répartition des classes (Pourcentage) :")
    repartition = df['label'].value_counts(normalize=True) * 100
    for classe, pourcentage in repartition.items():
        print(f"  - {classe:<25} : {pourcentage:.2f}%")
    print()
    
    # 4. Calcul de la longueur des textes en TOKENS (crucial pour max_length)
    print("Analyse de la longueur des textes (en tokens BERT)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenization de chaque texte pour obtenir la vraie longueur
    token_lengths = [len(tokenizer.encode(text, add_special_tokens=True)) for text in df['combined_text']]
    
    min_tokens = np.min(token_lengths)
    max_tokens = np.max(token_lengths)
    moyenne_tokens = np.mean(token_lengths)
    p95_tokens = np.percentile(token_lengths, 95) # 95% des textes font moins que cette taille
    
    print(f"  - Longueur minimale : {min_tokens} tokens")
    print(f"  - Longueur maximale : {max_tokens} tokens")
    print(f"  - Longueur moyenne  : {moyenne_tokens:.1f} tokens")
    print(f"  - 95ème percentile  : {p95_tokens:.1f} tokens (Idéal pour le choix de max_length)\n")
    
    # 5. Affichage de 5 exemples aléatoires avec leurs labels
    print("AFFICHAGE DE 5 EXEMPLES ALÉATOIRES :")
    exemples = df.sample(5, random_state=42) # random_state pour la reproductibilité
    
    for idx, row in exemples.iterrows():
        print(f"LABEL : {row['label']}")
        print(f"TEXTE : {row['combined_text'][:200]}...") # Tronqué à 200 car. pour l'affichage
        print("-" * 60)

if __name__ == "__main__":
    inspecter_dataset()