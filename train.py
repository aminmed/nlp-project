from models import *
import torch 
import torch.nn as nn 
import numpy as np 
import pandas as pd 
from transformers import  BertTokenizer, AdamW, get_linear_schedule_with_warmup
import os 
import argparse 
import configparser
from dataset import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning) 


SEED = 1234

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



# loss function is simple binary cross entropy loss
# need sigmoid to put probabilities in [0,1] interval
def loss_fn(outputs, targets):
    outputs = torch.squeeze(outputs)
    return nn.BCELoss()(nn.Sigmoid()(outputs), targets)


# computes perplexity on validation data
def calculate_perplexity(data_loader, model, device):
    model.eval()
    
    # tells Pytorch not to store values of intermediate computations for backward pass because we not gonna need gradients.
    with torch.no_grad():
        total_loss = 0
        for batch in data_loader:
            ids = batch["ids"].to(device, dtype=torch.long)
            mask = batch["mask"].to(device, dtype=torch.long)
            token_type_ids = batch["token_type_ids"].to(device, dtype=torch.long)
            targets = batch["targets"].to(device, dtype=torch.float)

            outputs = model(input_ids = ids, attention_mask=mask, token_type_ids=token_type_ids)
            total_loss += loss_fn(outputs, targets).item()
            
    model.train()

    return np.exp(total_loss / len(data_loader))

def train_loop(
        epochs,
        train_data_loader,
        val_data_loader,
        model,
        optimizer,
        device,
        scheduler=None,
        path_to_save_model= './checkpoints'
    ):
    it = 1
    total_loss = 0
    curr_perplexity = None
    perplexity = None
    
    epochs_ = tqdm(range(epochs), desc="Epochs", position=0, leave=True)
 
    model.train()

    for epoch in epochs_:



        for batch in tqdm(train_data_loader, desc="Training progress", position=0, leave=True):


            ids = batch["ids"].to(device, dtype=torch.long)
            mask = batch["mask"].to(device, dtype=torch.long)
            token_type_ids = batch["token_type_ids"].to(device, dtype=torch.long)
            targets = batch["targets"].to(device, dtype=torch.float)

            optimizer.zero_grad()
            
            # do forward pass, will save intermediate computations of the graph for later backprop use.
            outputs = model(input_ids = ids, attention_mask=mask, token_type_ids=token_type_ids)
            
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            
            # running backprop.
            loss.backward()
            
            # doing gradient descent step.
            optimizer.step()
            
            # we are logging current loss/perplexity in every 100 iteration
            if it % 100 == 0:
                
                # computing validation set perplexity in every 500 iteration.
                if it % 500 == 0:
                    curr_perplexity = calculate_perplexity(val_data_loader, model, device)
                    
                    if scheduler is not None:
                        scheduler.step()

                    # making checkpoint of best model weights.
                    if not perplexity or curr_perplexity < perplexity:
                        torch.save(model.state_dict(), path_to_save_model + 'ep_' + str(epoch))
                        perplexity = curr_perplexity

                print('\n | Iter', it, '| Avg Train Loss', total_loss / 100, '| Dev Perplexity', curr_perplexity)
                total_loss = 0

            it += 1
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='preprocessing of train and test dataset quora questions pairs detection')
    parser.add_argument('-c', '--configs', type=str, help='path to config file (hyper parameters)', default = './configs/config.ini')

    args = parser.parse_args() 
    config = configparser.ConfigParser()
    config.read(args.configs)
    
    ROOT_DATA         = config.get('Hyperparameters', 'ROOT_DATA')
    BERT_VERSION      = config.get('Hyperparameters', 'BERT_VERSION')
    POOLED_OUTPUT_DIM = config.getint('Hyperparameters', 'POOLED_OUTPUT_DIM')
    BATCH_SIZE        = config.getint('Hyperparameters', 'BATCH_SIZE')
    EPOCHS            = config.getint('Hyperparameters', 'EPOCHS')
    LR                = config.getfloat('Hyperparameters', 'LR')
    DROPOUT           = config.getfloat('Hyperparameters', 'DROPOUT')
    MLP_HEAD          = config.getboolean('Hyperparameters', 'MLP_HEAD')

    PATH_SAVE_MODEL   = './checkpoints/' + 'model_' + BERT_VERSION + '_' + str(BATCH_SIZE) + '_'


    tokenizer = BertTokenizer.from_pretrained(BERT_VERSION)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ### train and validation data loaders 
    train_df = pd.read_csv(os.path.join(ROOT_DATA , 'train.csv')).sample(10000)
    train_df, val_df = train_test_split(train_df, test_size=0.1)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_data_loader = get_data_loader(
    df=train_df,
    batch_size=BATCH_SIZE,
    shuffle=True,
    tokenizer=tokenizer
    )

    val_data_loader = get_data_loader(
    df=val_df,
    batch_size=4 * BATCH_SIZE,
    shuffle=True,
    tokenizer=tokenizer
    )

    # Training : 

    

    model = BertModel(
        BERT_VERSION,
        dropout_prob = DROPOUT,
        pooled_ouput_dim = POOLED_OUTPUT_DIM, 
        mlp_head = MLP_HEAD
    ).to(device)

    num_training_steps = int(len(train_data_loader) * EPOCHS)
    optimizer = AdamW(model.parameters(), lr=LR, no_deprecation_warning=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    
    train_loop(EPOCHS, train_data_loader, val_data_loader,  model, optimizer, device, scheduler, PATH_SAVE_MODEL)