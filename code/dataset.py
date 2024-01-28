
#This script is used to load the data and then merge the dataset.


import numpy as np
import pandas as pd
from ast import literal_eval

class Dataset:
    
    BASE_URL = "../input/nbme-score-clinical-patient-notes"

    @staticmethod
    def process_feature_text(text):
        return text.replace("-OR-", ";-").replace("-", " ")

    @staticmethod
    def prepare_datasets():
        features = pd.read_csv(f"{BASE_URL}/features.csv")
        notes = pd.read_csv(f"{BASE_URL}/patient_notes.csv")
        df = pd.read_csv(f"{BASE_URL}/train.csv")

        #literal_eval: turn string into list
        df["annotation_list"] = [literal_eval(x) for x in df["annotation"]]
        df["location_list"] = [literal_eval(x) for x in df["location"]]

        merged = df.merge(notes, how="left")
        merged = merged.merge(features, how="left")

        merged["feature_text"] = [process_feature_text(x) for x in merged["feature_text"]]
        merged["feature_text"] = merged["feature_text"].apply(lambda x: x.lower())
        merged["pn_history"] = merged["pn_history"].apply(lambda x: x.lower())

        return merged
    
    @staticmethod
    def create_test_df():
    feats = pd.read_csv(f"{BASE_URL}/features.csv")
    notes = pd.read_csv(f"{BASE_URL}/patient_notes.csv")
    test = pd.read_csv(f"{BASE_URL}/test.csv")

    merged = test.merge(notes, how = "left")
    merged = merged.merge(feats, how = "left")
    merged["feature_text"] = [process_feature_text(x) for x in merged["feature_text"]]
    
    return merged

