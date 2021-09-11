
import pandas as pd 
import numpy as np
from statistics import mean
import math
from datetime import datetime
import statistics as st
import datetime
import pickle
import time
import datetime
import sys


N_ESTIMATORS = 130000
MAX_DEPTH = 11
LEARNING_RATE = 0.2
MIN_DATA_IN_LEAF = 20

if len(sys.argv) == 5:
    N_ESTIMATORS = int(sys.argv[1])
    MAX_DEPTH = int(sys.argv[2])
    LEARNING_RATE = float(sys.argv[3])
    MIN_DATA_IN_LEAF = int(sys.argv[4])
    print(sys.argv[1:])

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

# Split the dataset into training_validation and testing part
# 80 : 20 

from sklearn.model_selection import train_test_split

validation_ratio = 0.2

X_train_valid, X_test_valid, y_train_valid, y_test_valid = train_test_split( 
    X_train, y_train,
    test_size = validation_ratio, 
    random_state = SEED
    )

X_train, X_test, y_train, y_test = \
    X_train_valid, X_test_valid, y_train_valid, y_test_valid

import lightgbm as lgb #pip3 install lightbm

# Instantiate a lgb.LGBMRegressor

lgbm = lgb.LGBMRegressor(
    seed=SEED,
    n_estimators = N_ESTIMATORS,
    max_depth = MAX_DEPTH,
    learning_rate = LEARNING_RATE,
    min_data_in_leaf = MIN_DATA_IN_LEAF
)

print("Starting fitting at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)
tic = time.perf_counter() #begin timing
#Fit with datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")SciKit
lgbm.fit(X_train, y_train)
time_fit_cv = time.perf_counter() - tic #save timer
print("Fitting completed at",
datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print()

print("Starting test set prediction at",
datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
tic = time.perf_counter() #begin timing
# Predict the test set labels 'y_pred0'
y_pred = lgbm.predict(X_test)
time_pred_cv = time.perf_counter() - tic #save timer
print("Test set prediction completed at",
datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print()

print("Starting train set prediction at",
datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
tic = time.perf_counter() #begin timing
# Predict the train set labels 'y_pred_train'
y_pred_train = lgbm.predict(X_train)
time_pred_train_cv = time.perf_counter() - tic #save timer
print("Train set prediction completed at",
datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print()

# Evaluate the test set MAPE
MAPE_test = mean_absolute_percentage_error(y_test, y_pred)

# Evaluate the train set MAPE
MAPE_train = mean_absolute_percentage_error(y_train, y_pred_train)

print("Time fit: ", time_fit_cv)
print()
print("MAPE train set: ", MAPE_train)
print("Time pred train set: ", time_pred_train_cv)
print("Row count train set: ", X_train.shape[0])
print()
print("MAPE test set: ", MAPE_test)
print("Time pred test set: ", time_pred_cv)
print("Row count test set: ", X_test.shape[0])
print()

print("Hyperparameters:")
print("N_ESTIMATORS:", N_ESTIMATORS)
print("MAX_DEPTH: ", MAX_DEPTH)
print("LEARNING_RATE: ", LEARNING_RATE)
print("MIN_DATA_IN_LEAF: ", MIN_DATA_IN_LEAF)
print()
