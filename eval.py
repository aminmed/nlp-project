import pandas as pd 
import numpy as np 
import torch 
import torch.nn as nn 

from dataset import get_data_loader
from tqdm import tqdm 
from transformers import  BertTokenizer
import xgboost as xgb

import os , pickle
import argparse
import configparser
from models import * 
from prettytable import PrettyTable


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)


def test(model, test_df, tokenizer,  device):
    
    predictions = torch.empty(0).to(device, dtype=torch.float)

    test_data_loader = get_data_loader(
        df = test_df, 
        batch_size= 512, 
        tokenizer=tokenizer,
        shuffle = False
    )
    
    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_data_loader):
            ids = batch["ids"]
            mask = batch["mask"]
            token_type_ids = batch["token_type_ids"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
  
            outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)

            predictions = torch.cat((predictions, nn.Sigmoid()(outputs)))
    
    return predictions.cpu().numpy().squeeze()




def eval(model, tokenizer, first_question, second_question, device):
    
    inputs = tokenizer.encode_plus(
        first_question,
        second_question,
        add_special_tokens=True,
    )

    ids = torch.tensor([inputs["input_ids"]], dtype=torch.long).to(device, dtype=torch.long)
    mask = torch.tensor([inputs["attention_mask"]], dtype=torch.long).to(device, dtype=torch.long)
    token_type_ids = torch.tensor([inputs["token_type_ids"]], dtype=torch.long).to(device, dtype=torch.long)

    with torch.no_grad():
        model.eval()
        output = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        prob = nn.Sigmoid()(output).item()

        print("questions [{}] and [{}] are {} with score {}".format(first_question, second_question, 'similar' if prob > 0.5 else 'not similar', prob))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Testing models')

    parser.add_argument(
        '--configs',
        type=str,
        help='path to test config file',
        default = './configs/bert_test_config.ini'
    )

    parser.add_argument(
                    '--model',
                    default='bert',
                    const='bert',
                    nargs='?',
                    choices=['bert', 'xgboost', 'ensemble'],
                    help='models: bert, xgboost or ensemble  (default: %(default)s)')

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.configs)
    PATH_DATA   = config.get('Hyperparameters', 'PATH_DATA')
    CHECKPOINT  = config.get('Hyperparameters', 'CHECKPOINT')

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    y = None 



    if args.model == 'bert' : 
        
        BERT_VERSION      = config.get('Hyperparameters', 'BERT_VERSION')
        pooled_output_dim = config.getint('Hyperparameters', 'POOLED_OUTPUT_DIM')
        dropout           = config.getfloat('Hyperparameters', 'DROPOUT')
        mlp_head          = config.getboolean('Hyperparameters', 'MLP_HEAD')

        print("validation data loading")
        df = pd.read_csv(os.path.join(ROOT_DATA , 'train_preprocessed.csv'))
        y = df['is_duplicate']
        print("data loading done.")

        tokenizer         = BertTokenizer.from_pretrained(BERT_VERSION)
        model = BertModel(BERT_VERSION,dropout,pooled_output_dim, mlp_head  ).to(device)
        model.load_state_dict(torch.load(CHECKPOINT,  map_location=device))
        predictions = test(model=model, test_df = df, tokenizer=tokenizer, device= device )


    elif args.model == "xgboost":

        print("validation data loading")
        df = pd.read_csv(os.path.join(PATH_DATA))
        df.drop(columns=['q1_glove', 'q2_glove'], axis = 1, inplace=True)
        y = pd.read_csv(os.path.join(ROOT_DATA , 'train_y.csv'))
        xgb_inputs = xgb.DMatrix(df)
        print("data loading done.")
        # Blank new instance to be loaded into
        xgboost_model = xgb.Booster()
        xgboost_model.load_model("./checkpoints/xgboost/model.json")
        predictions = xgboost_model.predict(xgb_inputs)
        predictions = torch.tensor(predictions, dtype=torch.float)

    else : 
        raise NotImplementedError("Only bert and xgboost evaluation   is implemented !!")

    y = torch.tensor(y.values, dtype=torch.float) 

    log_loss  = log_loss(y, predictions) 

    predictions = torch.where(predictions >= 0.5, 1, 0).type(torch.long)
    y = y.type(torch.long)

    accuracy  = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall    = recall_score(y, predictions)
    f1score   = f1_score(y, predictions) 

    
    x = PrettyTable()
    
    x.field_names = ["Metric ", "Value"]

    x.add_row(["Log loss",  log_loss.item() ])
    x.add_row(["Accuracy", accuracy ])
    x.add_row(["Recall", recall])
    x.add_row(["Precision", precision])
    x.add_row(["F1-score", f1score])
    print(x)



