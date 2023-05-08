import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import re 
import string 

nltk.download('stopwords')
nltk.download('wordnet')

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
    # Load the dataset from a CSV file
    df = pd.read_csv('quora_question_pairs.csv')

    # Apply the preprocessing function to the 'question1' and 'question2' columns
    df['question1_processed'] = df['question1'].apply(preprocess_text)
    df['question2_processed'] = df['question2'].apply(preprocess_text)

    # Save the preprocessed data to a new CSV file
    df.to_csv('quora_question_pairs_preprocessed.csv', index=False)
