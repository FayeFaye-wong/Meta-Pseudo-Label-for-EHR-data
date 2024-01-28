from dataset import Dataset
from CustomDataset import CustomDataset, preprocessTok
from trainer import CustomModel, ModelTrainer
from trainer_biLSTM import BiModel
from evaluation import Evaluation
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer 
from mpl import finetune
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import numpy as np


def main():
## SETTING HYPERPARAMETERS    
    hyperparameters = {
    "max_length": 416,
    "padding": "max_length",
    "return_offsets_mapping": True,
    "truncation": "only_second",
    "model_name": "../input/huggingface-bert/bert-base-uncased",
    "dropout": 0.2,
    "lr": 1e-5,
    "test_size": 0.2,
    "seed": 1268,
    "batch_size": 8
    }
    
## DATA PREPROCESSING
    # Load the data
    df = DataLoader.load_and_clean_data()
    #Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hyperparameters['model_name'])

    #Separate training set and testing set
    train_df = Dataset.prepare_datasets()
    X_train, X_test = train_test_split(train_df, test_size=hyperparameters['test_size'],
                                   random_state=hyperparameters['seed'])

    training_data = CustomDataset(X_train, tokenizer, hyperparameters)
    train_dataloader = DataLoader(training_data, batch_size=hyperparameters['batch_size'], shuffle=True)

    test_data = CustomDataset(X_test, tokenizer, hyperparameters)
    test_dataloader = DataLoader(test_data, batch_size=hyperparameters['batch_size'], shuffle=False)
    
        train_loss_data, test_loss_data = [], []
        
    score_data_list = []
    test_loss_min = np.Inf
    since = time.time()
    epochs = 3
    best_loss = np.inf
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = CustomModel(hyperparameters).to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=hyperparameters['lr'])
    
## Training Baseline Model
    for i in range(epochs):
        print("Epoch: {}/{}".format(i + 1, epochs))
        # first train model
        train_loss = train_model(model, train_dataloader, optimizer, criterion)
        train_loss_data.append(train_loss)
        print(f"Train loss: {train_loss}")
        # evaluate model
        test_loss, score = eval_model(model, test_dataloader, criterion)
        test_loss_data.append(test_loss)
        score_data_list.append(score)
        print(f"test loss: {test_loss}")
        print(f"test score: {score}")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), "nbme_bert_v2.pth")

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    plt.plot(train_loss_data, label="Training loss")
    plt.plot(test_loss_data, label="Testing loss")
    plt.legend(frameon=False)
    
    
    evaluation = Evaluation()

    # Call the eval_model method
    test_loss, score = evaluation.eval_model(model, dataloader, criterion)
     
    bimodel = BiModel(hyperparameters).to(DEVICE)
    bi_optimizer = optim.AdamW(bimodel.parameters(), lr=hyperparameters['lr'])

## Training BiLSTM Model
    for i in range(epochs):
        print("Epoch: {}/{}".format(i + 1, epochs))
        # first train model
        train_loss = train_model(bimodel, train_dataloader, bi_optimizer, criterion)
        train_loss_data.append(train_loss)
        print(f"Train loss: {train_loss}")
        # evaluate model
        test_loss, score = eval_model(model, test_dataloader, criterion)
        test_loss_data.append(test_loss)
        score_data_list.append(score)
        print(f"test loss: {test_loss}")
        print(f"test score: {score}")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), "nbme_bert_v2.pth")

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    plt.plot(train_loss_data, label="Training loss")
    plt.plot(test_loss_data, label="Testing loss")
    plt.legend(frameon=False)
    
    evaluation = Evaluation()

    # Call the eval_model method
    test_loss, score = evaluation.eval_model(bimodel, dataloader, criterion)
    
## FINETUNING BY MPL
    config = {
        "debug": False,
        "num_unlabelled": 10000, # number of unlabelled data points to be used for Meta Pseudo Labels
        "base_model_path": "../input/huggingface-bert/bert-base-uncased", # backbone
        "student_model_dir": "./trained_student", # save dir for student model
        "teacher_model_dir": "./trained_teacher", # save dir for teacher model
        "teacher_save_name": "teacher", # teacher checkpoint 
        "student_save_name": "student", # student checkpoint
        # save tokenized datasets here
        "output_dir": "./",
        "train_dataset_path": "train_dataset",
        "test_dataset_path": "test_dataset",
        "model_name": "../input/huggingface-bert/bert-base-uncased",
        }
    #Get unlabelled data
    train_patients = set(train_df["pn_num"].unique())
    unlabelled_notes_df = notes_df[~notes_df["pn_num"].isin(train_patients)].copy()
    unlabelled_df = pd.merge(unlabelled_notes_df, features_df, on=['case_num'], how='left')
    unlabelled_dataloader = DataLoader(unlabelled_data, batch_size=hyperparameters['batch_size'], shuffle=False)
    
    # Set the number of training steps
    num_steps = 1000  # Set the appropriate number of steps

    # Instantiate the finetune class
    finetune_instance = finetune(config, num_steps, train_dataloader, unlabelled_dataloader, test_dataloader)

    # Call the training loop
    finetune_instance.train()
    
    
    
    return

main()
