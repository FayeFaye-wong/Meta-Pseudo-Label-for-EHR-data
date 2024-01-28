
'''
Generate the features for Name Entity Recognization
'''
import numpy as np
import pandas as pd
from dataset import Dataset

class preprocessTok:
    #loc_list(location in type:list),turn list into int
    @staticmethod
    def loc_list_to_ints(loc_list):
        to_return = []
        for loc_str in loc_list:
            loc_strs = loc_str.split(";")
            for loc in loc_strs:
                start, end = loc.split()
                to_return.append((int(start), int(end)))
        return to_return

    #here we can understand in QA's way. Q is feature, the frist sentence,label=0,
    #A is named entity, in patient notes, label=1
    @staticmethod
    def tokenize_and_add_labels(tokenizer, data, config):
        out = tokenizer(
            data["feature_text"],
            data["pn_history"],
            truncation=config['truncation'],
            max_length=config['max_length'],
            padding=config['padding'],
            return_offsets_mapping=config['return_offsets_mapping']
        )
        labels = [0.0] * len(out["input_ids"])
        out["location_int"] = preprocessTok.loc_list_to_ints(data["location_list"])
        out["sequence_ids"] = out.sequence_ids()

        for idx, (seq_id, offsets) in enumerate(zip(out["sequence_ids"], out["offset_mapping"])):
            if not seq_id or seq_id == 0:
                labels[idx] = -1
                continue

            token_start, token_end = offsets
            for feature_start, feature_end in out["location_int"]:
                if token_start >= feature_start and token_end <= feature_end:
                    labels[idx] = 1.0
                    break

        out["labels"] = labels

        return out
    
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        tokens = preprocessTok.tokenize_and_add_labels(self.tokenizer, data, self.config)

        input_ids = np.array(tokens["input_ids"])
        attention_mask = np.array(tokens["attention_mask"])
        token_type_ids = np.array(tokens["token_type_ids"])

        labels = np.array(tokens["labels"])
        offset_mapping = np.array(tokens['offset_mapping'])
        sequence_ids = np.array(tokens['sequence_ids']).astype("float16")
        
        return input_ids, attention_mask, token_type_ids, labels, offset_mapping, sequence_ids

