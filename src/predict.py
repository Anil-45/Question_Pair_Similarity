"""Predict"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split 
from sklearn.metrics import log_loss
from feature_extraction import generate_features
from utils import load_dict

TEST_PATH = '../data/test.csv'
FINAL_TEST_PATH = '../data/final_test.csv'
PREDICTIONS_PATH = '../data/predictions.csv'
XGB_OPTIMAL_PATH="../models/xgb1.json"

def predict(X, proba=False):
    """Train

    Args:
        X (DataFrame): Features
        proba (bool): Returns probabilities if set to True.
    Returns:
        array: predictions
    """
    if not os.path.exists(XGB_OPTIMAL_PATH):
        print(f'{XGB_OPTIMAL_PATH} does not exist')
        pred = [-1 for _ in range(X.shape[0])]
        return np.array(pred)
        
    model = xgb.XGBClassifier()
    model.load_model(XGB_OPTIMAL_PATH)
    
    if proba:
        pred = model.predict_proba(X)
    else:
        pred = model.predict(X)
    
    return pred        

if "__main__" == __name__:
    if os.path.exists(FINAL_TEST_PATH):
        data = pd.read_csv(FINAL_TEST_PATH)
    else:
        data = pd.read_csv(TEST_PATH)
        data = generate_features(data, is_train=False)
        data.to_csv(FINAL_TEST_PATH, index=False)
    data['is_duplicate'] = predict(data.drop(columns=['id', 'is_duplicate']))
    data[['id', 'is_duplicate']].to_csv(PREDICTIONS_PATH, index=False)
    