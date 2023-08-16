"""Train"""

import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split 
from sklearn.metrics import log_loss
from feature_extraction import generate_features
from utils import load_dict

TRAIN_PATH = '../data/data.csv'
FINAL_DATA_PATH = '../data/final_data.csv'
BEST_PARAMS_PATH = '../models/best_config.pkl'
XGB_OPTIMAL_PATH="../models/xgboost_optimal.json"

def train(X, y):
    """Train

    Args:
        X (DataFrame): Features
        y (DataFrame): Target
    """
    if os.path.exists(BEST_PARAMS_PATH):
        params = load_dict(BEST_PARAMS_PATH)
    else:
        params = {}    
    
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size=0.2,
                                                      stratify=y,
                                                      shuffle=True,
                                                      random_state=12)
    
    params['scale_pos_weight'] = list(y_train).count(0)/list(y_train).count(1)
    params['objective'] = 'binary:logistic'
    
    model = xgb.XGBClassifier(**params, random_state=12)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    train_preds = model.predict_proba(X_train)[:, 1]
    train_loss = log_loss(y_train, train_preds)
    
    val_preds = model.predict_proba(X_val)[:, 1]
    val_loss = log_loss(y_val, val_preds)
    
    print("Train loss: ", train_loss)
    print("Val loss: ", val_loss)
    
    model.save_model(XGB_OPTIMAL_PATH)
    
if "__main__" == __name__:
    if os.path.exists(FINAL_DATA_PATH):
        data = pd.read_csv(FINAL_DATA_PATH)
    else:
        data = pd.read_csv(TRAIN_PATH)
        data = generate_features(data)
        data.to_csv(FINAL_DATA_PATH, index=False)
    train(data.drop(columns=['is_duplicate']), data['is_duplicate'])
    