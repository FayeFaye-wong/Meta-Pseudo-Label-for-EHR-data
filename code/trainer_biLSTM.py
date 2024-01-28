import time
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

class BiModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config['model_name'])  # BERT model
        self.lstm=nn.LSTM(input_size=768,hidden_size=384,batch_first=True,bidirectional=True)
        self.dropout = nn.Dropout(p=config['dropout'])
        self.config = config
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
        
    @staticmethod
    def forward(self, input_ids, attention_mask, token_type_ids):
        batch_size=input_ids.size(0)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        outputs,_=self.lstm(outputs)
        #outputs =self.dropout(outputs)  
        #outputs=outputs[:,-1,:]
        logits = self.fc1(outputs)
        #print(logits.size())
        logits = self.fc2(self.dropout(logits))
        logits = self.fc3(self.dropout(logits)).squeeze(-1)
        return logits
    


