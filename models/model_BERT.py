import torch
import torch.nn as nn
import torch.nn.functional as F 

import transformers
from transformers import BertModel 
transformers.logging.set_verbosity_error()

class BertModel(nn.Module):
    
    def __init__(self, bert_path, dropout_prob ,pooled_ouput_dim = 768,mlp_head = True):
        super(BertModel, self).__init__()
        
        self.bert_path = bert_path
        self.mlp_head = mlp_head
        self.dropout_prob = dropout_prob 
        self.pooled_ouput_dim = pooled_ouput_dim 
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)

        if mlp_head : 
            
            self.fc1 = nn.Linear(self.pooled_ouput_dim, self.pooled_ouput_dim // 4)
            self.fc2 = nn.Linear(self.pooled_ouput_dim // 4, 1)
            self.dropout = nn.Dropout(self.dropout_prob)
        
        else : 
            self.fc1 = nn.Linear(self.pooled_ouput_dim, 1)
            self.dropout = nn.Dropout(self.dropout_prob)


    def forward(self, input_ids, attention_mask, token_type_ids):
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = outputs.pooler_output

        if self.mlp_head : 

            x = F.relu(self.fc1(pooled_output))
            x = self.dropout(x)
            x = self.fc2(x)

        else : 
            x = self.dropout(pooled_output)
            x = self.fc1(x)
        
        return x



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


class LinearHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob):
        super(LinearHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):

        x = self.dropout(x)
        x = self.fc1(x)
        return x