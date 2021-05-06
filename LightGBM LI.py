
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
SEED = 500


from sklearn.metrics import mean_squared_error

import lightgbm as lgb #pip3 install lightbm

# Instantiate a lgb.LGBMRegressor
lgbm0 = lgb.LGBMRegressor(seed=SEED)

#Fit with SciKit
lgbm0.fit(X_train, y_train)

# Predict the test set labels 'y_pred0'
y_pred0 = lgbm0.predict(X_test)

# Evaluate the test set RMSE
rmse_test0 = mean_squared_error(y_test, y_pred0, squared=False)
print(rmse_test0)

########################
####Grid optimization###
########################

#setup params grid
param_grid = {'learning_rate': [0.01,0.2,0.5], #alias eta, Step size shrinkage used in update to prevents overfitting.  
    'n_estimators': [20, 50, 100],
    'max_depth': [3, 5, 10],
    'num_leaves': [16, 64, 128], 
    'min_data_in_leaf' : [16, 64, 128]
    }


from sklearn. model_selection import GridSearchCV
import time

#instantiate LGBMRegressor 
lgbm = lgb.LGBMRegressor(seed=SEED)
grid_mse = GridSearchCV(estimator=lgbm,
                        param_grid=param_grid,
                        scoring='neg_mean_squared_error', 
                        cv=3, 
                        verbose=1, 
                        n_jobs=1)
#fit  GridSearchCV 
tic = time.perf_counter() #begin timing
grid_mse.fit(X_train, y_train)
time_fit_cv = time.perf_counter() - tic #save timer

print("Best parameters found: ",grid_mse.best_params_) #best_params_
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_))) #best_score_

#extract the estimator best_estimator_ 
lgbm_ins = grid_mse.best_estimator_ #best_estimator_

# Predict the test set labels 'y_pred'
y_pred = lgbm_ins.predict(X_test)

# Evaluate the test set RMSE
rmse_test = mean_squared_error(y_test, y_pred, squared=False)
print(rmse_test)

#Evaluate 
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))


#send performance metrics to a google sheet,
#can be viewed at https://docs.google.com/spreadsheets/d/e/2PACX-1vSYyv4pRN7Q2EgDaGY7UGwpHCe6oN7fE3d951zaVKyi_Fh1S6gCGY9IY9dbQL4HqdW0wW3gGfGrGpLN/pubhtml 
#name to be filled in
import requests, datetime, json
requests.post(
    "https://sheet.best/api/sheets/6a3a81b3-be98-409b-9d40-8de4e0b3ee26",
    json={
        'Name': '____',
        'TEST': 'GRID',
        'RMSE': str(rmse_test),
        'DATETIME': datetime.datetime.now().isoformat(),
        'SEED': SEED,
        'RATIO': 20%,
        'PARAM_GRID': json.dumps(param_grid, indent=0),
        'BEST_PARAMS':json.dumps(grid_mse.best_params_, indent=0),
        'TIME_FIT': time_fit_cv,
        'LOW_RMSE': np.sqrt(np.abs(grid_mse.best_score_)),
        'RSCORE': r2_score(y_test, y_pred)
    },
)


###################################
####Grid randomized optimization###
###################################

from sklearn.model_selection import RandomizedSearchCV

#setup params grid
param_grid = {'learning_rate': [0.01,0.1,0.5], #alias eta, Step size shrinkage used in update to prevents overfitting.  
    'n_estimators': [20, 50, 100],
    'subsample': [0.5, 0.8, 1], #Subsample ratio of the training instances
    'max_depth': [3, 5, 10],
    'colsample_bytree': [0.5, 1] #colsample_bytree is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
    }

#instantiate LGBMRegressor 
lgbm = lgb.LGBMRegressor(seed=SEED)
grid_mse = RandomizedSearchCV(estimator=lgbm,
                        param_distributions=param_grid,
                        scoring='neg_mean_squared_error', 
                        cv=3, 
                        verbose=1, 
                        n_jobs=1)
#fit  GridSearchCV 
tic = time.perf_counter() #begin timing
grid_mse.fit(X_train, y_train)
time_fit_cv = time.perf_counter() - tic #save timer

print("Best parameters found: ",grid_mse.best_params_) #best_params_
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_))) #best_score_

#extract the estimator best_estimator_ 
lgbm_ins = grid_mse.best_estimator_ #best_estimator_

# Predict the test set labels 'y_pred'
y_pred = lgbm_ins.predict(X_test)

# Evaluate the test set RMSE
rmse_test = mean_squared_error(y_test, y_pred, squared=False)
print(rmse_test)

