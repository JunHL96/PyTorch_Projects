
# Write model.py file (this file loads the transformer model from its state_dict)

import torch
import torch.nn as nn
from transformers import DistilBertModel

class TransformerTextClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_prob=0.5):
        super(TransformerTextClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden_size = self.bert.config.hidden_size  # typically 768
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        normalized = self.batchnorm(cls_token)
        dropped = self.dropout(normalized)
        logits = self.classifier(dropped)
        return logits

def load_model(model_path: str, device: torch.device):
    model = TransformerTextClassifier(num_classes=2, dropout_prob=0.5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# End of model.py
