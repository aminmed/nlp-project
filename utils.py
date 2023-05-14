import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn 

from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
import scipy
from scipy.sparse import hstack
import os , pickle


from dataset import  get_data_loader 
from tqdm import tqdm 

MAX_FEATURES = 20000 # how many unique words to use (i.e num rows in embedding vector) 



def get_TFIDF_vectorization(data : pd.DataFrame):

    df = data.copy()

    #changing columns to numeric type
    num_cols = df.drop(columns=['id', 'qid1', 'qid2', 'question1', 'question2']).columns

    for col in num_cols:
        df[col] = df[col].apply(pd.to_numeric)
    
    # drop non-features columns
    to_drop_columns = ['id', 'qid1', 'qid2','is_duplicate']
    y = df['is_duplicate'] 
    X = df[df.drop(columns=to_drop_columns).columns.tolist()]

    # TFIDF computation for both question1 and question2
    
    tfidf_vectorizer_q1 = TfidfVectorizer(lowercase=False, max_features= MAX_FEATURES)
    q1_ifidf = tfidf_vectorizer_q1.fit_transform(X['question1'])

    tfidf_vectorizer_2 = TfidfVectorizer(lowercase=False,max_features= MAX_FEATURES)
    q2_ifidf = tfidf_vectorizer_2.fit_transform(X['question2'])

    tfidf_vec = hstack((q1_ifidf,q2_ifidf))

    df_X = X.drop(columns=['question1', 'question2'])

    
    df_X_sparse = scipy.sparse.csr_matrix(df_X)

    df = hstack((df_X_sparse,tfidf_vec))

    pickle.dump(df, open("./data/tfidf_X_tr","wb"))



def load_glove(word_index, embedding_file_path = '../embeddings/glove.840B.300d/glove.840B.300d.txt' ):

    """
    Loads pre-trained GloVe embeddings for a given vocabulary.

    Args:
    - word_index (dict): a dictionary or mapping that contains word-to-index
                         mappings for the vocabulary used in the text corpus. 
    - embedding_file_path (str): a string that specifies the file path of the pre-trained GloVe embeddings file.

    Returns:
    - embedding_matrix (np.array): a matrix of shape `(vocab_size, embedding_dim)` containing the GloVe embeddings for the vocabulary used in the text corpus.

    """
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]
    
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_file_path))
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.005838499,0.48782197
    embed_size = all_embs.shape[1]

    nb_words = min(MAX_FEATURES, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():
        if i >= MAX_FEATURES: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    return embedding_matrix 

    
def load_fasttext(word_index, embedding_file_path = '../embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec' ):    

    """
    Load pre-trained FastText embeddings for a given word index.

    Args:
        word_index (dict): A dictionary or mapping that contains word-to-index mappings for the vocabulary used in the text corpus.
                            The keys are the words, and the values are the corresponding integer indices.
        embedding_file_path (str): A string that specifies the file path of the pre-trained FastText embeddings file. 
                                    This should be a text file containing the embeddings in the format `<word> <embedding vector>`.

    Returns:
        embedding_matrix (np.array): a matrix of shape `(vocab_size, embedding_dim)` containing the FastText 
                                     embeddings for the vocabulary used in the text corpus. 
    """

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_file_path) if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(MAX_FEATURES, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    #embedding_matrix = np.random.normal(emb_mean, 0, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= MAX_FEATURES: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_para(word_index, embedding_file_path = '../embeddings/paragram_300_sl999/paragram_300_sl999.txt' ):
    
    """
    Load pre-trained ParaNMT-50 embeddings for a given word index.

    Args:
        word_index (dict): A dictionary or mapping that contains word-to-index mappings for the vocabulary used in the text corpus.
            The keys are the words, and the values are the corresponding integer indices.
        embedding_file_path (str): A string that specifies the file path of the pre-trained ParaNMT-50 embeddings file. 
            This should be a text file containing the embeddings in the format `<word> <embedding vector>`.

    Returns:

        embedding_matrix (np.array): a matrix of shape `(vocab_size, embedding_dim)` containing the FastText 
                                     embeddings for the vocabulary used in the text corpus.
    """
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_file_path, encoding="utf8", errors='ignore') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.0053247833,0.49346462
    embed_size = all_embs.shape[1]


    nb_words = min(MAX_FEATURES, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():
        if i >= MAX_FEATURES: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    return embedding_matrix



