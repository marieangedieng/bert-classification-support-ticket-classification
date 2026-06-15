import gradio as gr
import torch
import json
from transformers import AutoTokenizer, BertForSequenceClassification

# Chargement du dictionnaire de correspondance des étiquettes et du tokenizer associé
with open("label_mapping.json", "file" if not gr else "r") as f:
    mapping = json.load(f)

classes = mapping["classes"]
id2label = {int(k): v for k, v in mapping["id2label"].items()}
model_name = mapping["model_name"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained("best_bert_model")
model.eval()

def predict_ticket(subject, text):
    # Respect strict de la méthode d'agrégation d'entrée appliquée à l'entraînement
    combined_input = f"{subject} {text}"
    
    inputs = tokenizer(
        combined_input,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1).flatten()
        
    # Structuration du format attendu par Gradio (Label -> Probabilité)
    return {id2label[i]: float(probabilities[i]) for i in range(len(classes))}

# Construction de l'interface graphique
demo = gr.Interface(
    fn=predict_ticket,
    inputs=[
        gr.Textbox(label="Sujet du ticket (Subject)", placeholder="Ex: Problème d'accès VPN..."),
        gr.Textbox(label="Corps du message (Text)", placeholder="Ex: Bonjour, je ne parviens plus à me connecter depuis ce matin...", lines=5)
    ],
    outputs=gr.Label(num_top_classes=len(classes), label="Catégories prédites"),
    title="Système Intelligent de Classification de Tickets de Support (BERT)",
    description="Entrez le sujet et le contenu d'un ticket utilisateur pour prédire instantanément sa catégorie d'affectation automatique."
)

if __name__ == "__main__":
    demo.launch()