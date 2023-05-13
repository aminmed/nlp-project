

import torch
import xgboost as xgb
import numpy as np

class EnsembleModel:

    def __init__(self, neural_net_model_path, xgb_model_path):

        # loading pre-trained models : 

        self.neural_net = torch.load(neural_net_model_path)
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(xgb_model_path)

    def forward(self, input):
        pass 
    

    def predict(self, X):

        nn_inputs = torch.tensor(X).float()
        nn_outputs = self.neural_net(nn_inputs).detach().numpy()
        nn_probs = 1 / (1 + np.exp(-nn_outputs))
        xgb_inputs = xgb.DMatrix(X)
        xgb_probs = self.xgb_model.predict(xgb_inputs)
        ensemble_probs = 0.7*nn_probs + 0.3*xgb_probs  # Adjust weights to optimize performance
        ensemble_preds = np.where(ensemble_probs > 0.5, 1, 0)  # Convert probabilities to binary predictions
        return ensemble_preds


