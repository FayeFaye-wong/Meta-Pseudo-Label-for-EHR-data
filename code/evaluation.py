import numpy as np
from itertools import chain
import pandas as pd
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class Evaluation:
    @staticmethod
    def get_location_predictions(preds, offset_mapping, sequence_ids, test=False):
        all_predictions = []
        for pred, offsets, seq_ids in zip(preds, offset_mapping, sequence_ids):
            pred = 1 / (1 + np.exp(-pred))
            start_idx = None
            end_idx = None
            current_preds = []
            for pred, offset, seq_id in zip(pred, offsets, seq_ids):
                if seq_id is None or seq_id == 0:
                    continue

                if pred > 0.5:
                    if start_idx is None:
                        start_idx = offset[0]
                    end_idx = offset[1]
                elif start_idx is not None:
                    if test:
                        current_preds.append(f"{start_idx} {end_idx}")
                    else:
                        current_preds.append((start_idx, end_idx))
                    start_idx = None
            if test:
                all_predictions.append("; ".join(current_preds))
            else:
                all_predictions.append(current_preds)

        return all_predictions

    @staticmethod
    def calculate_char_cv(predictions, offset_mapping, sequence_ids, labels):
        all_labels = []
        all_preds = []
        for preds, offsets, seq_ids, labels in zip(predictions, offset_mapping, sequence_ids, labels):

            num_chars = max(list(chain(*offsets)))
            char_labels = np.zeros(num_chars)

            for o, s_id, label in zip(offsets, seq_ids, labels):
                if s_id is None or s_id == 0:
                    continue
                if int(label) == 1:
                    char_labels[o[0]:o[1]] = 1

            char_preds = np.zeros(num_chars)

            for start_idx, end_idx in preds:
                char_preds[start_idx:end_idx] = 1

            all_labels.extend(char_labels)
            all_preds.extend(char_preds)

        results = precision_recall_fscore_support(all_labels, all_preds, average="binary", labels=np.unique(all_preds))
        accuracy = accuracy_score(all_labels, all_preds)


        return {
            "Accuracy": accuracy,
            "precision": results[0],
            "recall": results[1],
            "f1": results[2]
        }
    @staticmethod
    def eval_model(model, dataloader, criterion):
        model.eval()
        valid_loss = []
        preds = []
        offsets = []
        seq_ids = []
        valid_labels = []

        for batch in tqdm(dataloader):
            input_ids = batch[0].to(DEVICE)
            attention_mask = batch[1].to(DEVICE)
            token_type_ids = batch[2].to(DEVICE)
            labels = batch[3].to(DEVICE)
            offset_mapping = batch[4]
            sequence_ids = batch[5]

            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            loss = torch.masked_select(loss, labels > -1.0).mean()
            valid_loss.append(loss.item() * input_ids.size(0))
            
            preds.append(logits.detach().cpu().numpy())
            offsets.append(offset_mapping.numpy())
            seq_ids.append(sequence_ids.numpy())
            valid_labels.append(labels.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        offsets = np.concatenate(offsets, axis=0)
        seq_ids = np.concatenate(seq_ids, axis=0)
        valid_labels = np.concatenate(valid_labels, axis=0)
        location_preds = get_location_predictions(preds, offsets, seq_ids, test=False)
        score = calculate_char_cv(location_preds, offsets, seq_ids, valid_labels)

        return sum(valid_loss)/len(valid_loss), score

