# Classification Automatique de Tickets de Support avec BERT

Par Marie-Ange DIENG, Christian MOUANGUE

Ce dépôt héberge l'implémentation complète d'un pipeline de Traitement Automatique du Langage Naturel (NLP) permettant de catégoriser des tickets de support client en exploitant un modèle de représentations de Transformers pré-entraîné (**BERT**).

## 1. Description de la Problématique et du Dataset

L'objectif est d'optimiser la file d'attente d'un support technique informatique en routant automatiquement chaque requête reçue vers le département adéquat.

Nous utilisons le dataset [support-ticket-classification.](https://drive.google.com/file/d/17QGX3a0x3iM1iGpd1RyBs1-Mhx2Ktckk/view)

Chaque observation se compose de deux variables d'entrée textuelles : le sujet (`subject`) et le corps du message (`text`). Pour maximiser l'extraction d'informations contextuelles par le mécanisme d'attention, le pipeline fusionne ces deux variables au format : `[Sujet] [Corps du message]`. Le modèle apprend à prédire la variable cible catégorielle `label` à partir de cette chaîne consolidée. `label` contient les classes : `account_access, billing, refund_request, bug_report` et `shipping_delivery`.
![Stats](images_rapport\stats.png)

Les classes sont bien équilibrées donc nous n'avons pas besoin de procéder à un rééquilibrage.

![Stats](images_rapport\exep.png)


## 2. Description du Modèle et Choix Techniques

### a. Architecture Globale du Modèle
Pour répondre à la tâche de classification des tickets de support, nous avons choisi l'architecture bert-base-uncased via la bibliothèque Hugging Face transformers.

Pourquoi BERT ? Contrairement aux modèles de langage traditionnels, BERT (Bidirectional Encoder Representations from Transformers) s'appuie sur un mécanisme d'attention bidirectionnel. Il analyse le contexte d'un mot à la fois à sa gauche et à sa droite, ce qui lui permet de capter le réel sens des mots et les subtilités d'une requête client.

Pourquoi la version uncased ? Les tickets de support contiennent fréquemment des irrégularités de casse (mots entièrement en majuscules pour exprimer une urgence, oublis de majuscules en début de phrase). Choisir une version uncased (insensible à la casse) permet de normaliser le texte et de réduire la taille du vocabulaire sans perte d'information sémantique pour cette tâche.

### b. Le Tokenizer et Prétraitement
Le prétraitement s'appuie sur le tokenizer officiel associé à bert-base-uncased, BertTokenizerFast ([docu auto](https://huggingface.co/docs/transformers/v5.12.0/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained))

Découpage en sous-mots : Ce tokenizer fragmente les mots inconnus ou complexes en sous-unités , ce qui résout efficacement le problème des mots hors-vocabulaire (OOV - Out-Of-Vocabulary) et gère bien les fautes de frappe des utilisateurs.

Jetons Spéciaux (Special Tokens) : 
* Le jeton [CLS] (Classification) est systématiquement inséré au tout début de chaque séquence textuelle.

* Le jeton [SEP] (Separation) est utilisé pour marquer la fin du texte.

Masque d'Attention (Attention Mask) : Le tokenizer génère un vecteur permettant au mécanisme d'attention de différencier les vrais tokens textuels des jetons de bourrage ([PAD]), évitant ainsi que le modèle ne calcule de l'attention sur du vide.

### c. Justification du max_length
Nous avons fixé la longueur maximale des séquences à max_length = 128.
Ce choix technique découle directement de l'Analyse Exploratoire des Données (EDA). En concaténant les champs subject (objet) et text (corps du ticket), l'analyse statistique montre que plus de 100 % des messages du dataset ont une longueur inférieure à 128 (en réalité même 85) tokens. 

### d. Tête de Classification
Pour adapter BERT (qui est initialement un modèle de masquage de langue) à notre tâche de classification à 5 classes, nous utilisons la classe BertForSequenceClassification. Le pipeline se déroule comme suit :
1. Extraction du pooling ([CLS]) : Le module récupere le vecteur d'état caché final correspondant uniquement au jeton initial [CLS].
2. Couche de régularisation (Dropout) : Une couche de Dropout (généralement fixée à $0.1$) est appliquée à ce vecteur pour désactiver aléatoirement certaines connexions pendant l'entraînement, ce qui prévient le surapprentissage (overfitting) et améliore la robustesse du modèle.
3. Couche Linéaire (Dense Layer) : Enfin, une couche linéaire (nn.Linear(768, 5)) projette ce vecteur de dimension 768 vers un espace de dimension 5, correspondant à nos 5 catégories de tickets de support (account_access, billing, bug_report, refund_request, shipping_delivery).
4. Fonction de perte : Les sorties brutes (logits) de cette couche linéaire sont directement passées à la fonction de perte CrossEntropyLoss de PyTorch pour l'entraînement, qui applique implicitement un Softmax pour calculer les probabilités par classe.
([docu BertForSequenceClassification](https://medium.com/@atri_iiita/fine-tuning-bert-for-text-classification-fcdc037637b5))

## 3. Détail des étapes de réalisation et difficultés rencontrées

### a. Analyse exploratoir
Via le script `stats.py` , on explore les données. Cette étape a permis de :
* Quantifier le volume total d'exemples et valider l'équilibre des 5 classes cibles (calcul des répartitions en pourcentage).
* Analyser la longueur des textes (en nombre de tokens).

### b. Pipeline de Données et Tokenisation
Pour garantir la reproductibilité du projet, un environnement virtuel isolé (.venv) a été configuré. Les dépendances critiques ont été consignées dans un fichier requirements.txt optimisé, intégrant les contraintes de versions minimales pour les paquets clés (torch>=2.4.0, transformers>=4.40.0, wandb, gradio) et pointant vers les index de téléchargement officiels de NVIDIA pour le support de l'accélération matérielle. Un fichier .gitignore strict a également été structuré pour exclure les fichiers lourds (poids des modèles, etc.) et les configurations locales.

### c. Pipeline de Données et Tokenisation
Mise en place de la classe Dataset personnalisée héritant de torch.utils.data.Dataset. Ce pipeline extrait et concatène les champs textuels (subject et text), puis applique la tokenisation. Le script génère ainsi les input_ids et les attention_mask nécessaires aux mécanismes d'attention du Transformer, avant de packager le tout dans des DataLoaders pour la gestion des mini-batchs.

### d. Instanciation du Modèle et Initialisation de l'Optimiseur
Chargement de l'architecture pré-entraînée bert-base-uncased via la classe BertForSequenceClassification. Cette opération substitue automatiquement la tête de classification d'origine par une couche linéaire finale dédiée à notre tâche à 5 classes. L'apprentissage a été configuré avec l'optimiseur standardisé AdamW de PyTorch combiné à un planificateur de taux d'apprentissage (learning rate scheduler).

### e. Boucle d'Entraînement et Suivi MLOps
Développement de la boucle principale d'entraînement (train_epoch) et de validation (eval_epoch). À chaque itération, les tenseurs sont envoyés sur le GPU, les gradients calculés par rétropropagation, et les poids mis à jour. L'intégralité des métriques (Loss, Accuracy, F1-score ) est synchronisée en temps réel sur la plateforme Weights & Biases (W&B) pour assurer un suivi rigoureux des courbes d'apprentissage et détecter d'éventuels phénomènes de surapprentissage (overfitting).

### f. Difficultés
1. Basculement intempestif sur l'exécution CPU (CPU Fallback) : Lors du premier lancement du script train.py, le modèle s'exécutait exclusivement sur le processeur (CPU), ignorant la carte graphique pourtant présente, ce qui se traduisait par des temps de calcul très longs. L'environnement a été nettoyé et réinstallé en forçant l'utilisation des dépôts officiels PyTorch pour CUDA via la commande :
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```
2. Évolution des signatures d'API de l'optimiseur AdamW: Le script d'entraînement plantait dès l'initialisation avec l'erreur : TypeError: AdamW.__init__() got an unexpected keyword argument 'correct_bias'. Le paramètre correct_bias=False, obsolète, a été purement et simplement retiré de l'appel constructeur dans train.py, permettant une instanciation fluide de l'optimiseur.

## 4. Résultats
### 1. Courbes de loss
![train_loss](images_rapport\train_loss.svg)
![val_loss](images_rapport\val_loss.svg)
### 2. Courbes de accuracy
![train_acc](images_rapport\train_acc.svg)
![val_acc](images_rapport\val_acc.svg)
### 3. Courbes de F1
![train_f1](images_rapport\train_f1.svg)
![val_f1](images_rapport\val_f1.svg)

### 4. Métriques finales
![Métriques finales](images_rapport\metriques_finales.png)

### 5. Matrice de confusion
![Matrice de confusion](images_rapport\cm.png)

### 6. Analyses
L'examen des courbes de convergence (Loss et Accuracy) met en évidence le comportement typique d'un fine-tuning de modèle de langue :

* Apprentissage rapide : Dès la première époque, le modèle atteint un palier de performance élevé, ce qui valide l'efficacité des représentations initiales de bert-base-uncased. Le modèle s'adapte très vite au language des tickets de support.
* Contrôle du surapprentissage (Overfitting) : On observe un léger écart (de l'ordre de 8%) entre l'exactitude d'entraînement (97.63%) et de validation (89.25%). De même, la perte de validation se stabilise autour de 0.40 tandis que la perte d'entraînement descend à 0.11. Bien qu'un début de surapprentissage soit perceptible, le score F1 en validation reste excellent (~89.2%), prouvant une solide capacité de généralisation sur des requêtes clients inédites.

L'examen de la matrice de confusion nous permet de discerner de plus près certaines tendances.

Le modèle excelle particulièrement sur deux catégories distinctes qui affichent des taux de réussite remarquables :
* account_access (77 / 80 classifications correctes) : Cette catégorie possède une signature lexicale très forte et exclusive (ex: "password", "login", "locked", "reset"), ce qui permet à BERT de l'isoler quasi parfaitement.
* shipping_delivery (76 / 80 classifications correctes) : Les requêtes liées à la logistique emploient un vocabulaire très spécifique ("tracking", "carrier", "address", "delivered") que le le modèle identifie.

L'analyse des erreurs révèle des proximités logiques mais problématiques :
* bug_report confondu avec account_access (9 erreurs) : C'est la confusion la plus fréquente du modèle. Elle s'explique par la nature des requêtes. Un client qui écrit "Je ne peux plus me connecter à l'application, l'écran se fige" décrit techniquement un bug, mais utilise le champ lexical de l'accès au compte.
* billing confondu avec account_access (6 erreurs) et refund_request (4 erreurs) :
Les requêtes de facturation partagent de nombreuses expressions avec les demandes de remboursement ("charge", "invoice", "credit card").
* refund_request confondu avec shipping_delivery (6 erreurs) :
Cette confusion provient des incidents de livraison. Un ticket formulé ainsi : "Mon colis est arrivé complètement déchiré, je veux être remboursé immédiatement" contient à la fois des marqueurs de livraison et de remboursement. Le modèle privilégie parfois le contexte logistique au détriment de l'intention finale de remboursement.

## 5. Démo Gradio
![Page de démo avant test](images_rapport\gradio_1.png)
![Page de démo test](images_rapport\gradio_2.png)

## 6. Instructions d'installation
### Prérequis

* **Système d'exploitation :** Windows 10 / 11
* **Python :** Version 3.10 à 3.13
* **Matériel (Optionnel mais recommandé) :** Une carte graphique NVIDIA avec des pilotes à jour pour l'accélération matérielle.

### Configuration de l'Environnement

Ouvrez votre terminal (PowerShell ou Invite de commandes) à la racine du projet, puis exécutez les étapes suivantes :

#### Étape 1.1 : Création de l'environnement virtuel
```bash
python -m venv .venv
```

#### Étape 1.2 : Activation de l'environnement virtuel
* **Sur PowerShell :**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
* **Sur l'Invite de commandes (cmd) :**
  ```cmd
  .venv\Scripts\activate.bat
  ```
> *Vous devez voir apparaître `(.venv)` au tout début de la ligne de votre terminal, confirmant que l'environnement est actif.*

#### Étape 1.3 : Mise à jour de `pip`
```bash
python -m pip install --upgrade pip
```

### Installation des Dépendances (Optimisé GPU/CUDA)

Pour exploiter votre carte graphique NVIDIA sous Windows et éviter les conflits de versions, l'installation de PyTorch doit pointer vers l'index officiel des binaires compilés pour CUDA.

#### Cas A : Si vous possédez un GPU NVIDIA (Recommandé)
Installez PyTorch configuré pour l'architecture **CUDA 12.4** (parfaitement stable avec les versions récentes de Python sous Windows) puis installez le reste des dépendances :
```bash
pip install torch --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
pip install -r requirements.txt
```

#### Cas B : Si vous n'avez pas de GPU (Exécution sur CPU)
```bash
pip install -r requirements.txt
```

### Lancement des Scripts

Chaque script doit être exécuté dans l'ordre chronologique du pipeline de Machine Learning. **Vérifiez que votre environnement virtuel `(.venv)` est toujours actif avant de lancer une commande.**

#### Script 1 : Analyse Exploratoire des Données (EDA)
Ce script calcule les statistiques du dataset, vérifie la répartition des classes, analyse la longueur des tokens BERT et affiche des exemples bruts.
```bash
python stats.py
```

#### Script 2 : Entraînement et Fine-Tuning de BERT
Ce script lance l'entraînement du modèle sur votre GPU (ou CPU) et calcule les métriques d'évaluation à chaque époque.
* **Note sur Weights & Biases (W&B) :** Au premier lancement, le terminal vous invitera à coller votre clé d'API W&B pour synchroniser vos courbes en direct. Si vous préférez exécuter l'entraînement en mode hors-ligne sans compte, lancez d'abord la commande `wandb offline` dans votre terminal.

```bash
python train.py
```
*Une fois l'entraînement terminé, le script génère la matrice de confusion finale et sauvegarde les poids optimisés du modèle dans le dossier local.*

#### Script 3 : Lancement de l'Interface Démo (Gradio)
Ce script charge le modèle fraîchement entraîné et déploie une interface graphique web interactive pour tester la classification en temps réel.
```bash
python demo.py
```

---

### Accès à l'Application Interactive

Dès que le script `demo.py` affiche le message de confirmation dans le terminal, ouvrez votre navigateur web et accédez à l'adresse locale suivante :

👉 **[http://127.0.0.1:7860](http://127.0.0.1:7860)**

Vous pouvez désormais saisir n'importe quel texte ou objet de ticket de support factice pour observer la prédiction du modèle ainsi que la distribution des probabilités calculée par la tête de classification BERT.

### Texte de test
Sujet: Can't access premium forum

Corps: I have upgraded my account but I still can't access premium forum even after relogin.



