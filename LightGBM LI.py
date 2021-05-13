
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


# TO DO variable importance


# Split the dataset into training_validation and testing part
# 95 : 5 

from sklearn.model_selection import train_test_split

validation_ratio = 0.05

X_train_valid, X_test_valid, y_train_valid, y_test_valid = train_test_split( 
    X_train, y_train,
    test_size = validation_ratio, 
    random_state = SEED
    )

X_train_valid, y_train_valid, X_test_valid, y_test_valid = X_train, y_train, X_test, y_test


# Setup params grid
# initial ranges
GRID_SIZE = 20
N_ESTIMATORS_MIN = 50
N_ESTIMATORS_MAX = 500
MAX_DEPTH_MIN = 3
MAX_DEPTH_MAX = 8
LEARNING_RATE_COEF_MIN = -3
LEARNING_RATE_COEF_MAX = -0.5
MIN_DATA_IN_LEAF_MIN = 10
MIN_DATA_IN_LEAF_MAX = 25

import random

SEED = 333

#random.seed(SEED) # DEACTIVATED
grid = pd.DataFrame({
    'n_estimators' : [random.randint(N_ESTIMATORS_MIN, N_ESTIMATORS_MAX) for x in range(GRID_SIZE)],
    'max_depth' : [random.randint(MAX_DEPTH_MIN, MAX_DEPTH_MAX) for x in range(GRID_SIZE)],
    'learning_rate' : np.power([10 for x in range(GRID_SIZE)], [random.uniform(LEARNING_RATE_COEF_MIN,
     LEARNING_RATE_COEF_MAX) for x in range(GRID_SIZE)]),
    'min_data_in_leaf' : [random.randint(MIN_DATA_IN_LEAF_MIN, MIN_DATA_IN_LEAF_MAX) for x in range(GRID_SIZE)]
    })



def fit_regressor(X_train, y_train, X_test, y_test, params):
    # Instantiate a lgb.LGBMRegressor
    lgbm0 = lgb.LGBMRegressor(seed=SEED,
    n_estimators=int(params['n_estimators']),
    max_depth=int(params['max_depth']),
    learning_rate=params['learning_rate'],
    min_data_in_leaf=int(params['min_data_in_leaf'])
    )
    #Fit with SciKit
    lgbm0.fit(X_train, y_train)
    # Predict the test set labels 'y_pred0'
    y_pred0 = lgbm0.predict(X_test)
    # Evaluate the test set RMSE
    MAPE_test0 = mean_absolute_percentage_error(y_test, y_pred0)
    return MAPE_test0, np.array_str(lgbm0.feature_importances_)




import time

# fit regressor and compute MAPE for each param vector
tic = time.perf_counter() #begin timing
MAPE_list = np.empty(grid.shape[0])
FIMP_list = np.empty(grid.shape[0], dtype=str)
for i in range(grid.shape[0]):
    MAPE_list[i], FIMP_list[i] = fit_regressor(
        X_train_valid, y_train_valid,
        X_test_valid, y_test_valid,
         grid.iloc[i]
        )
time_fit_cv = time.perf_counter() - tic #save timer
grid['MAPE'] = MAPE_list #add MAPE to grid  
grid['FIMP'] = FIMP_list #add FIMP to grid  

best_fit_no = np.argmin(grid['MAPE'])

low_MAPE = grid.iloc[best_fit_no]['MAPE']
best_params = grid.iloc[best_fit_no].drop('MAPE').drop('FIMP').astype(str).to_dict()
train_size = X_train_valid.shape[0]
columns = str(X_train_valid.columns.values)
fimp = grid.iloc[best_fit_no]['FIMP']

print(low_MAPE)


#send performance metrics to a google sheet,
#can be viewed at https://docs.google.com/spreadsheets/d/e/2PACX-1vSYyv4pRN7Q2EgDaGY7UGwpHCe6oN7fE3d951zaVKyi_Fh1S6gCGY9IY9dbQL4HqdW0wW3gGfGrGpLN/pubhtml 
#name to be filled in

NAME = '_________' # jmeno vyplnte sem

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
        'BEST_PARAMS': json.dumps(best_params, indent=0),
        'TIME_FIT': time_fit_cv,
        'LOW_RMSE': 'N/A',
        'MAPESCORE': low_MAPE,
        'N_ROWS_TRAIN': train_size,
        'GRID_SIZE': GRID_SIZE,
        'COLUMNS': columns,
        'FEATURE_IMPORTANCES': fimp
    }
)



#MAPE OBJ

def fit_regressor(X_train, y_train, X_test, y_test, params):
    # Instantiate a lgb.LGBMRegressor
    lgbm0 = lgb.LGBMRegressor(seed=SEED, objective='mape',
    n_estimators=int(params['n_estimators']),
    max_depth=int(params['max_depth']),
    learning_rate=params['learning_rate'],
    min_data_in_leaf=int(params['min_data_in_leaf'])
    )
    #Fit with SciKit
    lgbm0.fit(X_train, y_train)
    # Predict the test set labels 'y_pred0'
    y_pred0 = lgbm0.predict(X_test)
    # Evaluate the test set RMSE
    MAPE_test0 = mean_absolute_percentage_error(y_test, y_pred0)
    return MAPE_test0, np.array_str(lgbm0.feature_importances_)




import time

# fit regressor and compute MAPE for each param vector
tic = time.perf_counter() #begin timing
MAPE_list = np.empty(grid.shape[0])
FIMP_list = np.empty(grid.shape[0], dtype=str)
for i in range(grid.shape[0]):
    MAPE_list[i], FIMP_list[i] = fit_regressor(
        X_train_valid, y_train_valid,
        X_test_valid, y_test_valid,
         grid.iloc[i]
        )
time_fit_cv = time.perf_counter() - tic #save timer
grid['MAPE'] = MAPE_list #add MAPE to grid  
grid['FIMP'] = FIMP_list #add FIMP to grid  

best_fit_no = np.argmin(grid['MAPE'])

low_MAPE = grid.iloc[best_fit_no]['MAPE']
best_params = grid.iloc[best_fit_no].drop('MAPE').drop('FIMP').astype(str).to_dict()
train_size = X_train_valid.shape[0]
columns = str(X_train_valid.columns.values)
fimp = grid.iloc[best_fit_no]['FIMP']

print(low_MAPE)


#send performance metrics to a google sheet,
#can be viewed at https://docs.google.com/spreadsheets/d/e/2PACX-1vSYyv4pRN7Q2EgDaGY7UGwpHCe6oN7fE3d951zaVKyi_Fh1S6gCGY9IY9dbQL4HqdW0wW3gGfGrGpLN/pubhtml 

import requests, datetime, json
requests.post(
    "https://sheet.best/api/sheets/6a3a81b3-be98-409b-9d40-8de4e0b3ee26",
    json={
       'Name': NAME,
        'TEST': 'VECTOR MAPE OBJ',
        'RMSE': 'N/A',
        'DATETIME': datetime.datetime.now().isoformat(),
        'SEED': 'inactive',
        'RATIO': 'N/A',
        'PARAM_GRID': 'N/A',
        'BEST_PARAMS': json.dumps(best_params, indent=0),
        'TIME_FIT': time_fit_cv,
        'LOW_RMSE': 'N/A',
        'MAPESCORE': low_MAPE,
        'N_ROWS_TRAIN': train_size,
        'GRID_SIZE': GRID_SIZE,
        'COLUMNS': columns,
        'FEATURE_IMPORTANCES': fimp
    }
)
