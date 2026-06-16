# Classification Automatique de Tickets de Support avec BERT

Par Marie-Ange DIENG, Christian MOUANGUE

Ce dépôt héberge l'implémentation complète d'un pipeline de Traitement Automatique du Langage Naturel (NLP) permettant de catégoriser des tickets de support client en exploitant un modèle de représentations de Transformers pré-entraîné (**BERT**).

## 1. Description de la Problématique et du Dataset

L'objectif est d'optimiser la file d'attente d'un support technique informatique en routant automatiquement chaque requête reçue vers le département adéquat.

Nous utilisons le dataset [support-ticket-classification.](https://drive.google.com/file/d/17QGX3a0x3iM1iGpd1RyBs1-Mhx2Ktckk/view)

Chaque observation se compose de deux variables d'entrée textuelles : le sujet (`subject`) et le corps du message (`text`). Pour maximiser l'extraction d'informations contextuelles par le mécanisme d'attention, le pipeline fusionne ces deux variables au format : `[Sujet] [Corps du message]`. Le modèle apprend à prédire la variable cible catégorielle `label` à partir de cette chaîne consolidée. `label` contient les classes : `account_access, billing, refund_request, bug_report` et `shipping_delivery`.


## 2. Architecture Implémentée

Nous exploitons l'architecture **BERT-Base-Uncased** (110 Millions de paramètres) configurée en mode Fine-tuning pour la classification de séquences (`BertForSequenceClassification`).
Le modèle est entraîné de bout en bout en adaptant l'ensemble de ses poids, tout en appliquant des hyperparamètres prévenant le phénomène d'oubli catastrophique (*catastrophic forgetting*).

## 3. Instructions d'Exécution

### Installation des dépendances

```bash
pip install -r requirements.txt
```


