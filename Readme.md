# Quora question pair similarity


Our implementation is based on Python and is inspired by the following articles:

- https://arxiv.org/pdf/1907.01041.pdf
- https://aclanthology.org/P19-1465/

and these githubs repositories: 
- https://github.com/YuriyGuts/kaggle-quora-question-pairs

## Details of the used approach:
Our approach involves preprocessing the question pairs, extracting features, developing a BERT-based model for similarity prediction as well as a classical machine learning model, evaluating its performance using metrics, and iteratively improving the model based on error analysis.

## Dependencies:
 ````
 numpy, nltk, pandas, transformers, pytorch, sklearn 
 ```` 
 
## Run the code:
### Preprosseing
### How to train ?

	

### How to test ?

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

  

