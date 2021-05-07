
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

#setup params grid

import random

GRID_SIZE = 20
SEED = 333

random.seed(SEED)
grid = pd.DataFrame({
    'n_estimators' : [random.randint(50, 500) for x in range(GRID_SIZE)],
    'max_depth' : [random.randint(3, 8) for x in range(GRID_SIZE)],
    'learning_rate' : np.power([10 for x in range(GRID_SIZE)], [random.uniform(-3, -0.5) for x in range(GRID_SIZE)]),
    'min_data_in_leaf' : [random.randint(10, 25) for x in range(GRID_SIZE)]
    })


def fit_regressor(X_train, y_train, params):
    # Instantiate a lgb.LGBMRegressor
    lgbm0 = lgb.LGBMRegressor(seed=SEED,
    #n_estimators=params['n_estimators'],
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
    
    return {'MAPE': MAPE_test0}


# fit regressor and compute MAPE for each param vector
MAPE_list = np.empty(grid.shape[0])
for i in range(grid.shape[0]):
    MAPE_list[i] = fit_regressor(
        X_train_valid, y_train_valid, grid.iloc[i]
        )['MAPE']

#add MAPE to grid  
grid['MAPE'] = MAPE_list


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
                        scoring=mape_scorer, 
                        cv=3, 
                        verbose=1, 
                        n_jobs=1)
#fit  GridSearchCV 
tic = time.perf_counter() #begin timing
grid_mse.fit(X_train, y_train)
time_fit_cv = time.perf_counter() - tic #save timer

print("Best parameters found: ",grid_mse.best_params_) #best_params_
print("Lowest MAPE found: ", np.abs(grid_mse.best_score_)) #best_score_

#extract the estimator best_estimator_ 
lgbm_ins = grid_mse.best_estimator_ #best_estimator_

# Predict the test set labels 'y_pred'
y_pred = lgbm_ins.predict(X_test)

# Evaluate the test set R2score
mape_test = mean_absolute_percentage_error(y_test, y_pred)
print(mape_test)


#send performance metrics to a google sheet,
#can be viewed at https://docs.google.com/spreadsheets/d/e/2PACX-1vSYyv4pRN7Q2EgDaGY7UGwpHCe6oN7fE3d951zaVKyi_Fh1S6gCGY9IY9dbQL4HqdW0wW3gGfGrGpLN/pubhtml 
#name to be filled in
import requests, datetime, json
requests.post(
    "https://sheet.best/api/sheets/6a3a81b3-be98-409b-9d40-8de4e0b3ee26",
    json={
        'Name': '____',
        'TEST': 'GRID',
        'RMSE': 'N/A',
        'DATETIME': datetime.datetime.now().isoformat(),
        'SEED': SEED,
        'RATIO': 'N/A',
        'PARAM_GRID': json.dumps(param_grid, indent=0),
        'BEST_PARAMS':json.dumps(grid_mse.best_params_, indent=0),
        'TIME_FIT': time_fit_cv,
        'LOW_RMSE': 'N/A',
        'MAPESCORE': mean_absolute_percentage_error(y_test, y_pred)
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

