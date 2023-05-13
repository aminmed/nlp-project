import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn 


from dataset import  get_data_loader 
from tqdm import tqdm 

max_features = 120000 # how many unique words to use (i.e num rows in embedding vector) 


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

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():
        if i >= max_features: continue
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
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    #embedding_matrix = np.random.normal(emb_mean, 0, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
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


    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    return embedding_matrix



# this function returns probabilities for every test case.
def test(model, test_df, tokenizer,  device):
    predictions = torch.empty(0).to(device, dtype=torch.float)
    

    test_data_loader = get_data_loader(
        df = test_df, 
        batch_size= 512, 
        tokenizer=tokenizer
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

            outputs = model(ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
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

        