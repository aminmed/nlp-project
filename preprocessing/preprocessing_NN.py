import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import glob 
import re 
import string 

# progress bar ...
from tqdm import tqdm 


import os 
import argparse 


tqdm.pandas() 
#nltk.download('stopwords')
#nltk.download('wordnet')



def preprocess_text(text):
    # Convert text to lowercase

    text = text.lower()


    # Remove punctuation
    #text = ''.join(c for c in text if c not in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Expand contractions
    contraction_patterns = [
        (r"won\'t", "will not"),
        (r"can\'t", "cannot"),
        (r"n\'t", " not"),
        (r"\'re", " are"),
        (r"\'s", " is"),
        (r"\'d", " would"),
        (r"\'ll", " will"),
        (r"\'t", " not"),
        (r"\'ve", " have"),
        (r"\'m", " am")
    ]
    for pattern in contraction_patterns:
        text = re.sub(pattern[0], pattern[1], text)


    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.casefold() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word) for word in filtered_words]

    # Tokenization
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(' '.join(lemmas))

    return ' '.join(tokens)


if __name__ == '__main__': 


    print("Start preprocessing for BERT embeddings")

    # Load the dataset from a CSV file
    
    parser = argparse.ArgumentParser(description='preprocessing of train and test dataset quora questions pairs detection')
    parser.add_argument('-r', '--root', type=str, help='path to root folder of data', default = './data')

    args = parser.parse_args() 


    if len(glob.glob(args.root + '/*_NN_preprocessed.csv' )) != 0 :

        print("preprocessed csv files already exists, aborting ...")
        
        exit() 
        
    path_train = os.path.join(args.root, 'train.csv')
    path_test = os.path.join(args.root, 'test.csv')



    df_train = pd.read_csv(path_train)

    df_train.dropna(inplace=True) 
    print("Preprocessing and extracting features for training dataset")
    # Apply the preprocessing function to the 'question1' and 'question2' columns
    df_train['question1_processed'] = df_train['question1'].progress_apply(preprocess_text)
    df_train['question2_processed'] = df_train['question2'].progress_apply(preprocess_text)

    # Save the preprocessed data to a new CSV file
    df_train.to_csv(os.path.join(args.root, 'train_NN_preprocessed.csv'), index=False)

    df_test = pd.read_csv(path_test)

    df_test.dropna(inplace=True) 
    print("Preprocessing and extracting features for testing dataset")
    # progress_apply the preprocessing function to the 'question1' and 'question2' columns
    df_test['question1_processed'] = df_test['question1'].progress_apply(preprocess_text)
    df_test['question2_processed'] = df_test['question2'].progress_apply(preprocess_text)

    # Save the preprocessed data to a new CSV file
    df_test.to_csv(os.path.join(args.root, 'test_NN_preprocessed.csv'), index=False)

    print("Done")