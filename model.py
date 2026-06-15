from transformers import BertForSequenceClassification

def get_bert_model(model_name="bert-base-uncased", num_labels=2, label2id=None, id2label=None):
    """Instancie le modèle BERT adapté au nombre de classes ciblé."""
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label
    )
    return model