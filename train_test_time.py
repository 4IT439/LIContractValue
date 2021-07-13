
import pandas as pd 
import numpy as np
from statistics import mean
import math
from datetime import datetime
import statistics as st
import datetime
import pickle
import time

N_ESTIMATORS = 200
MAX_DEPTH = 3
LEARNING_RATE = 0.1
MIN_DATA_IN_LEAF = 20

#Load the file containing variables [X_train, y_train, X_test, y_test]
import pickle
with open(r"LLcvalue.pickle", "rb") as input_file:
    X_train, y_train, X_test, y_test = pickle.load(input_file)

#%reset -f
SEED = 333

from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

import lightgbm as lgb #pip3 install lightbm

# Instantiate a lgb.LGBMRegressor

lgbm = lgb.LGBMRegressor(
    seed=SEED,
    n_estimators = N_ESTIMATORS,
    max_depth = MAX_DEPTH,
    learning_rate = LEARNING_RATE,
    min_data_in_leaf = MIN_DATA_IN_LEAF
)


tic = time.perf_counter() #begin timing

#Fit with SciKit
lgbm.fit(X_train, y_train)

time_fit_cv = time.perf_counter() - tic #save timer


tic = time.perf_counter() #begin timing
# Predict the test set labels 'y_pred0'
y_pred = lgbm.predict(X_test)

time_pred_cv = time.perf_counter() - tic #save timer


tic = time.perf_counter() #begin timing
# Predict the train set labels 'y_pred_train'
y_pred_train = lgbm.predict(X_train)

time_pred_train_cv = time.perf_counter() - tic #save timer


# Evaluate the test set MAPE
MAPE_test = mean_absolute_percentage_error(y_test, y_pred)

# Evaluate the train set MAPE
MAPE_train = mean_absolute_percentage_error(y_train, y_pred_train)

print("Time fit: ", time_fit_cv)
print()
print("MAPE train set: ", MAPE_train)
print("Time pred train set: ", time_pred_train_cv)
print("No rows train set: ", X_train.shape[0])
print()
print("MAPE test set: ", MAPE_test)
print("Time pred test set: ", time_pred_cv)
print("No rows test set: ", X_test.shape[0])

#send performance metrics to a google sheet,
#can be viewed at https://docs.google.com/spreadsheets/d/e/2PACX-1vSYyv4pRN7Q2EgDaGY7UGwpHCe6oN7fE3d951zaVKyi_Fh1S6gCGY9IY9dbQL4HqdW0wW3gGfGrGpLN/pubhtml 
#name to be filled in

NAME = '_____' # jmeno vyplnte sem

import requests, datetime, json
requests.post(
    "https://sheet.best/api/sheets/6a3a81b3-be98-409b-9d40-8de4e0b3ee26",
    json={
       'Name': NAME,
        'TEST': 'VECTOR',
        'RMSE': 'N/A',
        'DATETIME': datetime.datetime.now().isoformat(),
        'SEED': 'inactive',
        'RATIO': 'N/A',
        'PARAM_GRID': 'N/A',
        'R2SCORE': 'N/A',
        'BEST_PARAMS': json.dumps(best_params, indent=0),
        'TIME_FIT': time_fit_cv,
        'LOW_RMSE': 'N/A',
        'MAPESCORE': low_MAPE,
        'N_ROWS_TRAIN': train_size,
        'GRID_SIZE': GRID_SIZE,
        'COLUMNS': columns,
        'FEATURE_IMPORTANCES': fimp,
        'MAPE_TEST_SET' : MAPE_test_set,
        '2_ND_BEST_MAPE' : sec_best_MAPE,
        '2_ND_BEST_PARAMS' : json.dumps(sec_best_params, indent=0),
        'MAPE_TRAIN_SET' : MAPE_train_set
    }
)
