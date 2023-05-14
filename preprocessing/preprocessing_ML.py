import pandas as pd



# progress bar ...
from tqdm import tqdm 

import glob
import os , pickle
import argparse

from utils import get_TFIDF_GLOVE_vectorization, get_TFIDF_vectorization, extract_features


tqdm.pandas() 

if __name__ == '__main__': 


    print("Start preprocessing for classical ML models")

    # Load the dataset from a CSV file
    
    parser = argparse.ArgumentParser(description='preprocessing of train and test dataset quora questions pairs detection')
    parser.add_argument('-r', '--root', type=str, help='path to root folder of data', default = './data')
    parser.add_argument('-g','--glove' ,action='store_true', help='flag to use glove with IFIDF')

    args = parser.parse_args() 


    # if len(glob.glob(args.root + '/*_ML_preprocessed.csv' )) != 0 :

    #     print("preprocessed csv files already exists, aborting ...")
        
    #     exit() 
        
    path_train = os.path.join(args.root, 'train.csv')


    df_train = pd.read_csv(path_train)

    df_train.dropna(inplace=True) 

    print("Preprocessing and extracting features for training dataset")
    df_train = extract_features(df_train)

    print("Start vectorization : ")
    if args.glove : 
        df, y = get_TFIDF_GLOVE_vectorization(df_train)
        df.to_csv('./data/train_data.csv',index=False)
    else: 
        df, y = get_TFIDF_vectorization(df_train)
        pickle.dump(df, open("./data/tfidf_X_tr","wb"))
    

    # Save the y column to a new CSV file
    y.to_csv(os.path.join(args.root, 'train_y.csv'), index=False)

    #df_test = pd.read_csv(path_test)
    # df_test.dropna(inplace=True) 
    # # progress_apply the preprocessing the test df 
    # print("Preprocessing and extracting features for testing dataset")
    # df_test = extract_features(df_test)
    # # Save the preprocessed data to a new CSV file
    # df_test.to_csv(os.path.join(args.root, 'test_ML_preprocessed.csv'), index=False)

    print("Done")

