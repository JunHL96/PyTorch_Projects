import torch
import torch.nn as nn
from transformers import DistilBertModel

class TransformerTextClassifier(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        super(TransformerTextClassifier, self).__init__()
        # Load pretrained DistilBERT model
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden_size = self.bert.config.hidden_size  # typically 768
        # Batch normalization applied on the CLS token
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Expects x to be a dict with keys 'input_ids' and 'attention_mask'.
        """
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the first token ([CLS]) representation for classification
        cls_token = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        normalized = self.batchnorm(cls_token)
        dropped = self.dropout(normalized)
        logits = self.classifier(dropped)
        return logits
