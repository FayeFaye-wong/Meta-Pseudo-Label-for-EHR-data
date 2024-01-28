import time
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

class CustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config['model_name'])  # BERT model
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
    
class ModelTrainer:
    
    @staticmethod
    def train_model(model, dataloader, optimizer, criterion):
            model.train()
            train_loss = []

            for batch in tqdm(dataloader):
                optimizer.zero_grad()
                input_ids = batch[0].to(DEVICE)
                attention_mask = batch[1].to(DEVICE)
                token_type_ids = batch[2].to(DEVICE)
                labels = batch[3].to(DEVICE)

                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels)

                loss = torch.masked_select(loss, labels > -1.0).mean()
                train_loss.append(loss.item() * input_ids.size(0))
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            return sum(train_loss)/len(train_loss)
    
    @staticmethod
    def get_predictions(model, dataloader):
        model.eval()
        predictions = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids = batch[0].to(DEVICE)
                attention_mask = batch[1].to(DEVICE)
                token_type_ids = batch[2].to(DEVICE)

                logits = model(input_ids, attention_mask, token_type_ids)
                probabilities = torch.sigmoid(logits)
                predictions.extend(probabilities.cpu().numpy())

        return predictions
        

