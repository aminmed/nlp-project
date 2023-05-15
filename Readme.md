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
!pip install transformers, nltk, tokenizers, spacy, fuzzywuzzy, distance, python-levenshtein, prettytable

!python -m spacy download en_core_web_sm

 ```` 
 
## Run the code:
To use our implementation, follow these steps:
### Data Exploration:
To discover the dataset, simply run the notebook ./notebooks/Exploratory Data Analysis.ipynb. 
### Preprocessing 
Preprocess the raw question pairs by running the preprocessing code. This will transform the questions into a suitable format for feature extraction and model training. make sure your ./data folder looks like this : 

````
data/
├── train.csv
└── test.csv
 ````

##### Preprocessing for Classical ML models 

````
#Use the flag `-g` to indicate to the script 
#to apply TFIDF Weighted Glove Vectors 
#otherwise apply only TFIDF vetorization

python ./preprocessing/preprocessing_ML.py --root ./data/ -g
python ./preprocessing/preprocessing_ML.py --root ./data/
````
The output will be in data folder with the fowllowing structure : 

````
data/
├── train.csv
├── test.csv
├── train_y.csv
├── train_data.csv
└── tfidf_X_tr.csv
````

##### Preprocessing for BertModel 

````
python ./preprocessing/preprocessing_NN.py --root ./data/ 

````
The output will be in data folder with the fowllowing structure :  

````
data/
├── train.csv
├── test.csv
├── train_NN_preprocessed.csv
└── test_NN_preprocessed.csv
````

### How to train ?
Train the model by executing the training code. This will train the model using the preprocessed data and the defined model architecture. 

#### Train BertModel : 
to train the model that use BERT embeddings, run the following :

````
python train.py --config ./configs/config.ini
````

the configs folder containes files of configs where you can set hyperparameters and paths to data ...

#### Train classical ML models : 
To train classical ML models, simply run the notebook ./notebooks/ML_models.ipynb. 
Make sure you have already run preprocessing for ML models. 



### How to test ?
Test the trained model on new question pairs by running the testing code. This will provide predictions on the similarity of the question pairs based on the trained model. to evaluate the trained models, edit the config file in ./configs (example : ./configs/bert_test_config.ini) than run the following command : 
 ````
 python eval.py --configs ./configs/bert_test_config.ini --model bert

 ````

## Results
Our approach achieves promising results in the task of quora question pair similarity. The model achieves high accuracy and demonstrates good performance across various evaluation metrics. In the following, the detailed results in term of log loss: 
| ML Model | Vectorizer | Train log loss | Test log loss |
| --- | --- | --- | --- |

| Finetuned BERT |   --   | 0.13857 | 0.20404 |
| XGBoost | TF-IDF w2v | 0.21667 | 0.31909 |
| Linear SVM | TF-IDF w2v  | 0.38734 |  0.39245  |
| Logistic Regression | -- | 0.38432 |  0.39038 |

## Contributors :
  - DJECTA Hibat_Errahmen
  - BENCHEIKH LEHOCINE Mohammed Amine
  - BAZ Roland


  

  

