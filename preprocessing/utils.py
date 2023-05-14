import numpy as np 
import pandas as pd 

from sklearn.feature_extraction.text import TfidfVectorizer

import scipy
from scipy.sparse import hstack
from fuzzywuzzy import fuzz
import spacy
import distance
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import re 

from bs4 import BeautifulSoup

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

    return df, y 
    


def get_TFIDF_GLOVE_vectorization(data : pd.DataFrame):
    
    df = data.copy()
    #changing columns to numeric type
    num_cols = df.drop(columns=['id', 'qid1', 'qid2', 'question1', 'question2']).columns

    for col in num_cols:
        df[col] = df[col].apply(pd.to_numeric)
    
    # drop non-features columns
    to_drop_columns = ['id', 'qid1', 'qid2','is_duplicate']
    y = df['is_duplicate'] 
    X = df[df.drop(columns=to_drop_columns).columns.tolist()]


    questions = list(df['question1']) + list(df['question2'])
    tfidf = TfidfVectorizer(lowercase=False)
    tfidf.fit_transform(questions)
    # dict key:word and value:tf-idf score
    word2tfidf = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

    nlp = spacy.load('en_core_web_sm')

    vecs1 = []
    # https://github.com/noamraph/tqdm
    # tqdm is used to print the progress bar
    for qu1 in tqdm(list(X['question1'])):
        
        doc1 = nlp(qu1) 
        # 384 is the number of dimensions of vectors 
        mean_vec1 = np.zeros([len(doc1), 96])
        
        for i,word1 in enumerate(doc1):
            # word2vec
            vec1 = word1.vector
            # fetch df score
            try:
                idf = word2tfidf[str(word1)]
            except:
                idf = 0
            # compute final vec
            mean_vec1[i] += vec1 * idf
    
        mean_vec1 = mean_vec1.mean(axis=0)
        vecs1.append(mean_vec1)

    df_glove_q1 = vecs1


    vecs2 = []
    # https://github.com/noamraph/tqdm
    # tqdm is used to print the progress bar
    for qu2 in tqdm(list(X['question2'])):
        
        doc2 = nlp(qu2) 
        # 384 is the number of dimensions of vectors 
        mean_vec2 = np.zeros([len(doc2), 96])
        
        for i,word2 in enumerate(doc2):
            # word2vec
            vec2 = word2.vector
            # fetch df score
            try:
                idf = word2tfidf[str(word2)]
            except:
                idf = 0
            # compute final vec
            mean_vec2[i] += vec2 * idf
    
        mean_vec2 = mean_vec2.mean(axis=0)
        vecs2.append(mean_vec2)

    df_glove_q2 = vecs2

    X['q1_glove'], X['q2_glove'] = df_glove_q1, df_glove_q2

    np_glove = np.concatenate([np.array(df_glove_q1),np.array(df_glove_q2)],axis=1)
    
    df_glove = pd.DataFrame(np_glove,columns=[f'g_{i}' for i in range(np_glove.shape[1])])

    X = X.drop(columns=['question1','question2']).reset_index(drop=True)
    X = pd.concat([X,df_glove],axis=1)


    return X, y


def preprocess_text(text ):

    text = str(text).lower()
    text = text.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will")
    
    #replacing multiple digits representation to  miilion,thoudsands etc.. eg:1000 -> 1k
    text = re.sub(r"([0-9]+)000", r"\1k", text)
    text = re.sub(r"([0-9]+)000000", r"\1m", text)  
    
    
    porter = PorterStemmer()    #apply stemming  eg: growing,growth --> grow
    pattern = re.compile('\W')  #matching word charecter
    
    if type(text) == type(''):
        text = re.sub(pattern, ' ', text)
    
    
    if type(text) == type(''):
        text = porter.stem(text)
        example1 = BeautifulSoup(text, features="html.parser")
        text = example1.get_text()
               
    
    return text


def common_word(row):
    x = set(row['question1'].lower().strip().split(" ")) 
    y = set(row['question2'].lower().strip().split(" "))
    return 1.0 * len(x & y)


def total(row):
    set1 = set(row['question1'].lower().strip().split(" "))
    set2 = set(row['question2'].lower().strip().split(" "))
    return 1.0 * (len(set1) + len(set2))

def word_share(row):
    x = row['word_common']/row['word_total']
    return  x

# get the Longest Common sub string

def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))      # will return longest common substring 
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)



def get_token_features(q1, q2):
    
    token_features = [0.0]*10
    STOP_WORDS = set(stopwords.words('english'))
    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    #Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))
    
    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    SAFE_DIV = 0.0001 
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    
    #Average Token Length of both Questions
    token_features[9] = (len(q1_tokens) + len(q2_tokens))/2

    return token_features


  
def extract_features(data : pd.DataFrame): 
    
    df = data.copy()

    # preprocessing each question
    df["question1"] = df["question1"].fillna("").apply(preprocess_text)
    df["question2"] = df["question2"].fillna("").apply(preprocess_text)

    print("token features ...")

    token_features = df.apply(lambda x: get_token_features(x["question1"], x["question2"]), axis=1)
    
    df["cwc_min"]       = list(map(lambda x: x[0], token_features))
    df["cwc_max"]       = list(map(lambda x: x[1], token_features))
    df["csc_min"]       = list(map(lambda x: x[2], token_features))
    df["csc_max"]       = list(map(lambda x: x[3], token_features))
    df["ctc_min"]       = list(map(lambda x: x[4], token_features))
    df["ctc_max"]       = list(map(lambda x: x[5], token_features))
    df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
    df["first_word_eq"] = list(map(lambda x: x[7], token_features))
    df["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
    df["mean_len"]      = list(map(lambda x: x[9], token_features))

    # computing frequencies of questions according to ids 
    print("statistics features ...")
    qids = pd.Series(df.qid2.tolist() + df.qid1.tolist()) 
    cnt = qids.value_counts() 

    df['freq_qid1'] = df['qid1'].apply(lambda x: cnt[x])
    df['freq_qid2'] = df['qid2'].apply(lambda x: cnt[x])

    # length of both questions 
    df['q1len'] = df['question1'].apply(lambda x: len(x))
    df['q2len'] = df['question2'].apply(lambda x: len(x))

    # number of words in each question 
    df['q1_n_words'] = df['question1'].apply(lambda x: len(x.split(" ")))
    df['q2_n_words'] = df['question2'].apply(lambda x: len(x.split(" ")))
    
    # number of common words between question1 and question2 
    df['word_common'] = df.apply(common_word,axis=1)
    # total number of words 
    df['word_total'] = df.apply(total,axis=1)
    # rate of common words to the total 
    df['word_share'] = df.apply(word_share,axis=1)
    # sum of frequencies of both questions 
    df['freq_q1+q2'] = df['freq_qid1']+df['freq_qid2']
    # difference of frequencies of both questions 
    df['freq_q1-q2'] = abs(df['freq_qid1']-df['freq_qid2'])

    #Computing Fuzzy Features and Merging with Dataset
    
    # do read this blog: http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
    # https://stackoverflow.com/questions/31806695/when-to-use-which-fuzz-function-to-compare-2-strings
    # https://github.com/seatgeek/fuzzywuzzy
    print("fuzzy features ...")

    df["token_set_ratio"]       = df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
    # The token sort approach involves tokenizing the string in question, sorting the tokens alphabetically, and 
    # then joining them back into a string We then compare the transformed strings with a simple ratio().
    df["token_sort_ratio"]      = df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
    df["fuzz_ratio"]            = df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
    df["fuzz_partial_ratio"]    = df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
    df["longest_substr_ratio"]  = df.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)
    print("done")
    
    return df 