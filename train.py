from models import *
import torch 
import torch.nn as nn 
import numpy as np 
import pandas as pd 
from transformers import  BertTokenizer, AdamW, get_linear_schedule_with_warmup
import os 
import argparse 
from dataset import *
from sklearn.model_selection import train_test_split


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

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            total_loss += loss_fn(outputs, targets).item()
            
    model.train()

    return np.exp(total_loss / len(data_loader))

def train_loop(epochs, train_data_loader, val_data_loader, model, optimizer, device, scheduler=None):
    it = 1
    total_loss = 0
    curr_perplexity = None
    perplexity = None
    
    model.train()
    for epoch in range(epochs):
        print('Epoch: ', epoch + 1)
        for batch in train_data_loader:
            ids = batch["ids"].to(device, dtype=torch.long)
            mask = batch["mask"].to(device, dtype=torch.long)
            token_type_ids = batch["token_type_ids"].to(device, dtype=torch.long)
            targets = batch["targets"].to(device, dtype=torch.float)

            optimizer.zero_grad()
            
            # do forward pass, will save intermediate computations of the graph for later backprop use.
            outputs = model(ids, mask=mask, token_type_ids=token_type_ids)
            
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
                        torch.save(model.state_dict(), 'saved_model')
                        perplexity = curr_perplexity

                print('| Iter', it, '| Avg Train Loss', total_loss / 100, '| Dev Perplexity', curr_perplexity)
                total_loss = 0

            it += 1
        


if __name__ == "__main__":
    
    ROOT_DATA = "./data/"
    BERT_VERSION = 'bert-base-uncased'
    POOLED_OUTPUT_DIM = 768 
    BATCH_SIZE = 128
    EPOCHS = 1 
    LR = 3e-5


    tokenizer = BertTokenizer.from_pretrained(BERT_VERSION)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ### train and validation data loaders 
    train_df = pd.read_csv(os.path.join(ROOT_DATA , 'train.csv'))
    train_df, val_df = train_test_split(train_df, test_size=0.1)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_data_loader = get_data_loader(
    df=train_df,
    targets=train_df["is_duplicate"].values,
    batch_size=BATCH_SIZE,
    shuffle=True,
    tokenizer=tokenizer
    )

    val_data_loader = get_data_loader(
    df=val_df,
    targets=val_df["is_duplicate"].values,
    batch_size=4 * BATCH_SIZE,
    shuffle=True,
    tokenizer=tokenizer
    )

    # Training : 

    
    mlp_head = MLPHead(
        input_dim=POOLED_OUTPUT_DIM,
        hidden_dim=POOLED_OUTPUT_DIM,
        output_dim= 1
    )
    model = BertModel(BERT_VERSION, classification_head = mlp_head).to(device)

    num_training_steps = int(len(train_data_loader) * EPOCHS)
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    
    train_loop(EPOCHS, train_data_loader, val_data_loader,  model, optimizer, device, scheduler)