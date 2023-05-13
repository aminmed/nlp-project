import torch
import torch.nn as nn
import torch.nn.functional as F 

import transformers
from transformers import BertModel 

class BertModel(nn.Module):
    
    def __init__(self, bert_path, classification_head):
        super(BertModel, self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.classification_head = classification_head

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        logits = self.classification_head(pooled_output)
        return logits



class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x