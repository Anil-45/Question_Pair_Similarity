"""Hyperparameter Optimization."""

import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from utils import save_dict

FINAL_DATA_PATH = '../data/final_data.csv'
BEST_PARAMS_PATH = '../models/best_config.pkl'

def objective(trial, data, target):
    """Objective function.

    Args:
        trial (trial): trial
        data (DataFrame): data
        target (DataFrame): target

    Returns:
        float: log loss
    """
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=77)
    y_preds = np.zeros(target.shape[0])
    scores = []
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.30, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "max_depth": trial.suggest_int("max_depth", 3, 9, step=2),
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 100, 500),
        "n_estimators": trial.suggest_int("n_estimators", 4000, 7000, step=1000),
        'objective': 'binary:logistic'
    }
      
    for fold, (idx_train, idx_valid) in enumerate(kf.split(data, target)):
        train_x, train_y = data.iloc[idx_train], target.iloc[idx_train]
        test_x, test_y = data.iloc[idx_valid], target.iloc[idx_valid]
        
        model = xgb.XGBClassifier(**params, random_state=121,
                scale_pos_weight=list(train_y).count(0)/list(train_y).count(1))

        model.fit(train_x,train_y, eval_set=[(test_x,test_y)],verbose=False)

        preds = model.predict_proba(test_x)[:, 1]
        y_preds[idx_valid] = preds
        score = log_loss(test_y, preds)
        scores.append(score)
        print(f"fold : {fold+1} loss : ", score)
        

    print(f"average loss: {np.mean(scores)}")
    score = log_loss(target, y_preds)
    print("overall loss: ", score)
    return score

def optimize(X, y, n_trials):
    """Optimize

    Args:
        X (DataFrame): data
        y (DataFrame): target
        n_trials (int): Number of trials

    Returns:
        dict: Best Parameters
    """
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    return study.best_trial.params

if "__main__" == __name__:
    data = pd.read_csv(FINAL_DATA_PATH)
    X = data.drop(columns=['is_duplicate'])
    y = data['is_duplicate']
    best_params = optimize(X, y, n_trials=5)
    save_dict(best_params, BEST_PARAMS_PATH)