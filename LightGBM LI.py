#!/usr/bin/env python3
import pandas as pd 
import numpy as np
from statistics import mean
import math
from datetime import datetime
import statistics as st
import datetime
import pickle


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

mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

import lightgbm as lgb #pip3 install lightbm


# Instantiate a lgb.LGBMRegressor
lgbm0 = lgb.LGBMRegressor(seed=SEED)

#Fit with SciKit
lgbm0.fit(X_train, y_train)

# Predict the test set labels 'y_pred0'
y_pred0 = lgbm0.predict(X_test)

# Evaluate the test set RMSE
MAPE_test0 = mean_absolute_percentage_error(y_test, y_pred0)
print(MAPE_test0)


#################################
##### Stepwise Optimization #####
#################################


# Split the dataset into training_validation and testing part
# 80 : 20 

from sklearn.model_selection import train_test_split

validation_ratio = 0.2

X_train_valid, X_test_valid, y_train_valid, y_test_valid = train_test_split( 
    X_train, y_train,
    test_size = validation_ratio, 
    random_state = SEED
    )

# Setup params grid
# initial ranges
GRID_SIZE = 5
N_ESTIMATORS_MIN = 9000
N_ESTIMATORS_MAX = 20000
MAX_DEPTH_MIN = 4
MAX_DEPTH_MAX = 12
LEARNING_RATE_COEF_MIN = -1
LEARNING_RATE_COEF_MAX = -0.5
MIN_DATA_IN_LEAF_MIN = 20
MIN_DATA_IN_LEAF_MAX = 20
LEARNING_RATE_EXPL = 0 # keep 0 here, otherwise LEARNING_RATE_COEF will be omitted

import random

SEED = 333

#random.seed(SEED) # DEACTIVATED
grid = pd.DataFrame({
    'n_estimators' : [random.randint(N_ESTIMATORS_MIN, N_ESTIMATORS_MAX) for x in range(GRID_SIZE)],
    'max_depth' : [random.randint(MAX_DEPTH_MIN, MAX_DEPTH_MAX) for x in range(GRID_SIZE)],
    'learning_rate' : np.power([10 for x in range(GRID_SIZE)], [random.uniform(LEARNING_RATE_COEF_MIN,
     LEARNING_RATE_COEF_MAX) for x in range(GRID_SIZE)]) if LEARNING_RATE_EXPL==0 else [LEARNING_RATE_EXPL for x in range(GRID_SIZE)],
    'min_data_in_leaf' : [random.randint(MIN_DATA_IN_LEAF_MIN, MIN_DATA_IN_LEAF_MAX) for x in range(GRID_SIZE)]
    })


def fit_regressor(X_train, y_train, X_test, y_test, params):
    # Instantiate a lgb.LGBMRegressor
    lgbm = lgb.LGBMRegressor(seed=SEED,
    n_estimators=int(params['n_estimators']),
    max_depth=int(params['max_depth']),
    learning_rate=params['learning_rate'],
    min_data_in_leaf=int(params['min_data_in_leaf'])
    )
    #Fit with SciKit
    lgbm.fit(X_train, y_train)
    # Predict the test set labels 'y_pred0'
    y_pred0 = lgbm.predict(X_test)
    # Evaluate the test set RMSE
    MAPE_test0 = mean_absolute_percentage_error(y_test, y_pred0)
    return MAPE_test0, str(tuple(lgbm0.feature_importances_)), lgbm


import time

# fit regressor and compute MAPE for each param vector
tic = time.perf_counter() #begin timing
MAPE_list = np.empty(grid.shape[0])
FIMP_list = ['' for x in range(grid.shape[0])]
lgbm_list = ['' for x in range(grid.shape[0])]
for i in range(grid.shape[0]):
    MAPE_list[i], FIMP_list[i], lgbm_list[i]  = fit_regressor(
        X_train_valid, y_train_valid,
        X_test_valid, y_test_valid,
         grid.iloc[i]
        )
time_fit_cv = time.perf_counter() - tic #save timer
grid['MAPE'] = MAPE_list #add MAPE to grid  
grid['FIMP'] = FIMP_list #add FIMP to grid  


best_fit_order = np.argpartition(grid['MAPE'], 1)
best_fit_no = np.argmin(grid['MAPE'])
sec_best_fit_no = best_fit_order[1]

low_MAPE = grid.iloc[best_fit_no]['MAPE']
best_params = grid.iloc[best_fit_no].drop('MAPE').drop('FIMP').astype(str).to_dict()
train_size = X_train_valid.shape[0]
v_set_ratio = X_test_valid.shape[0] / (X_train_valid.shape[0] + X_test_valid.shape[0])
t_set_ratio = X_test.shape[0] / (X_train.shape[0] + X_test.shape[0])
columns = str(X_train_valid.columns.values)
fimp = grid.iloc[best_fit_no]['FIMP']

y_pred = lgbm_list[best_fit_no].predict(X_test)
MAPE_test_set = mean_absolute_percentage_error(y_test, y_pred)

# sec_best_params = grid.iloc[sec_best_fit_no].drop('MAPE').drop('FIMP').astype(str).to_dict()
sec_best_params = 'N/A'

# y_pred = lgbm_list[sec_best_fit_no].predict(X_test_valid)
# sec_best_MAPE = mean_absolute_percentage_error(y_test_valid, y_pred)
sec_best_MAPE = 'N/A'

# y_pred = lgbm_list[best_fit_no].predict(X_train_valid)
# MAPE_train_set = mean_absolute_percentage_error(y_train_valid, y_pred)
MAPE_train_set = 'N/A'

print(low_MAPE)
print(best_params)
print(FIMP_list[best_fit_no])


#send performance metrics to a google sheet,
#can be viewed at https://docs.google.com/spreadsheets/d/e/2PACX-1vSYyv4pRN7Q2EgDaGY7UGwpHCe6oN7fE3d951zaVKyi_Fh1S6gCGY9IY9dbQL4HqdW0wW3gGfGrGpLN/pubhtml 
#name to be filled in

NAME = '_____' # jmeno vyplnte sem

import requests, datetime, json
requests.post(
    "https://sheet.best/api/sheets/6a3a81b3-be98-409b-9d40-8de4e0b3ee26",
    json={
        'NAME': NAME,
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
        'V_SET_RATIO': v_set_ratio,
        'T_SET_RATIO': t_set_ratio,
        'GRID_SIZE': GRID_SIZE,
        'COLUMNS': columns,
        'FEATURE_IMPORTANCES': fimp,
        'MAPE_TEST_SET' : MAPE_test_set,
        '2_ND_BEST_MAPE' : sec_best_MAPE,
        '2_ND_BEST_PARAMS' : json.dumps(sec_best_params, indent=0),
        'MAPE_TRAIN_SET' : MAPE_train_set
    }
)

