# Quora question pair similarity


Our implementation is based on Python and is inspired by the following articles:

- https://arxiv.org/pdf/1907.01041.pdf
- https://aclanthology.org/P19-1465/

and these githubs repositories: 
- https://github.com/YuriyGuts/kaggle-quora-question-pairs

## Details of the used approach:
Our approach involves preprocessing the question pairs, extracting features, developing a BERT-based model for similarity prediction as well as a classical machine learning model, evaluating its performance using metrics, and iteratively improving the model based on error analysis. More details in the report joined to this repository.

## Installing dependencies:

#### Conda envs : 

````
#Create a conda environment
conda create --name <environment-name> python=3.8.* 
#Activate the created conda environment
conda activate <environment-name>
#install dependencies from requirements.txt 
conda install --file requirements.txt

 ```` 
#### Other (colab) : 
Using pip commands 

````
!pip install transformers, nltk, tokenizers, spacy, fuzzywuzzy, distance, python-levenshtein

!python -m spacy download en_core_web_sm

 ```` 
 
## Run the code:
To use our implementation, follow these steps:
### Preprosseing
Preprocess the raw question pairs by running the preprocessing code. This will transform the questions into a suitable format for feature extraction and model training.
 ````

 ````
### How to train ?
Train the model by executing the training code. This will train the model using the preprocessed data and the defined model architecture.
 ````

 ````
### How to test ?
Test the trained model on new question pairs by running the testing code. This will provide predictions on the similarity of the question pairs based on the trained model.
 ````

 ````
## Results
Our approach achieves promising results in the task of quora question pair similarity. The model achieves high accuracy and demonstrates good performance across various evaluation metrics. In the following, the detailed results in term of log loss: 
| ML Model | Vectorizer | Train log loss | Test log loss |
| --- | --- | --- | --- |
| XGBoost | TF-IDF w2v | 0.21667200942242115 | 0.31909646297475824 |
| BERT | TF-IDF w2v | 0.21667200942242115 | 0.31909646297475824 |
## Contributors :
  - BAZ Roland

  - BENCHEIKH LEHOCINE Mohammed Amine
  
  - DJECTA Hibat_Errahmen

  

