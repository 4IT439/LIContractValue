<<<<<<< HEAD

import pandas as pd 
import numpy as np
from statistics import mean
import math
from datetime import datetime
import statistics as st
import datetime
import pickle
<<<<<<< HEAD
=======


# load all files
df1 = pd.read_csv('scen_0001-0200.csv').iloc[:1000]
df2 = pd.read_csv('scen_0201-0500.csv').iloc[:1000]
df3 = pd.read_csv('scen_0501-0700.csv').iloc[:1000]
df4 = pd.read_csv('scen_0701-1000.csv').iloc[:1000]
df5 = pd.read_csv('scen_1001-1300.csv').iloc[:1000]
df6 = pd.read_csv('scen_1301-1600.csv').iloc[:1000]
df7 = pd.read_csv('scen_1601-1900.csv').iloc[:1000]
df8 = pd.read_csv('scen_1901-2236.csv').iloc[:1000]
df9 = pd.read_csv('scen_6001-6300.csv').iloc[:1000]
df10 = pd.read_csv('scen_6301-6600.csv').iloc[:1000]
df11 = pd.read_csv('scen_6601-6900.csv').iloc[:1000]
df12 = pd.read_csv('scen_6901-7200.csv').iloc[:1000]
df13 = pd.read_csv('scen_7201-7500.csv').iloc[:1000]
df14 = pd.read_csv('scen_7501-7800.csv').iloc[:1000]
df15 = pd.read_csv('scen_7801-8236.csv').iloc[:1000]
df16 = pd.read_csv('scen_9001-9300.csv').iloc[:1000]
df17 = pd.read_csv('scen_9301-9700.csv').iloc[:1000]
df18 = pd.read_csv('scen_9701-10000.csv').iloc[:1000]
df19 = pd.read_csv('scen_10001-10300.csv').iloc[:1000]
df20 = pd.read_csv('scen_10301-10600.csv').iloc[:1000]
#df21 = pd.read_csv('./dataset/scen_10601-10900.csv').iloc[:1000]
#df22 = pd.read_csv('./dataset/scen_10901-11236.csv').iloc[:1000]
>>>>>>> 5213d3cfa5d76ee5b4d1b8eab94bcc3099e1a767


# load all files and get random intances
RATIO = 0.0001 # portion of dataset

def df_get_instances(df, ratio):
    df = df.iloc[np.random.randint(0, df.shape[0], int(df.shape[0] * ratio))]
    return df

df1 = pd.read_csv('scen_0001-0200.csv')
df1 = df_get_instances(df1, RATIO)

df2 = pd.read_csv('scen_0201-0500.csv')
df2 = df_get_instances(df2, RATIO)

df3 = pd.read_csv('scen_0501-0700.csv')
df3 = df_get_instances(df3, RATIO)

df4 = pd.read_csv('scen_0701-1000.csv')
df4 = df_get_instances(df4, RATIO)

df5 = pd.read_csv('scen_1001-1300.csv')
df5 = df_get_instances(df5, RATIO)

df6 = pd.read_csv('scen_1301-1600.csv')
df6 = df_get_instances(df6, RATIO)

df7 = pd.read_csv('scen_1601-1900.csv')
df7 = df_get_instances(df7, RATIO)

df8 = pd.read_csv('scen_1901-2236.csv')
df8 = df_get_instances(df8, RATIO)

df9 = pd.read_csv('scen_6001-6300.csv')
df9 = df_get_instances(df9, RATIO)

df10 = pd.read_csv('scen_6301-6600.csv')
df10 = df_get_instances(df10, RATIO)

df11 = pd.read_csv('scen_6601-6900.csv')
df11 = df_get_instances(df11, RATIO)

df12 = pd.read_csv('scen_6901-7200.csv')
df12 = df_get_instances(df12, RATIO)

df13 = pd.read_csv('scen_7201-7500.csv')
df13 = df_get_instances(df13, RATIO)

df14 = pd.read_csv('scen_7501-7800.csv')
df14 = df_get_instances(df14, RATIO)

df15 = pd.read_csv('scen_7801-8236.csv')
df15 = df_get_instances(df15, RATIO)

df16 = pd.read_csv('scen_9001-9300.csv')
df16 = df_get_instances(df16, RATIO)

df17 = pd.read_csv('scen_9301-9700.csv')
df17 = df_get_instances(df17, RATIO)

df18 = pd.read_csv('scen_9701-10000.csv')
df18 = df_get_instances(df18, RATIO)

df19 = pd.read_csv('scen_10001-10300.csv')
df19 = df_get_instances(df19, RATIO)

df20 = pd.read_csv('scen_10301-10600.csv')
df20 = df_get_instances(df20, RATIO)

df21 = pd.read_csv('scen_10601-10900.csv')
df21 = df_get_instances(df21, RATIO)

df22 = pd.read_csv('scen_10901-11236.csv')
df22 = df_get_instances(df22, RATIO)

#load curves following NSS optimization
dfnss = pd.read_csv('nss_tau2.csv')
dfnss.drop(['Unnamed: 0'], axis='columns', inplace=True)

# join them together
df_total = pd.concat([eval('df'+str(i+1)) for i in range(22)])

# join them together, ___ rows
#df_total = pd.concat([eval('df'+str(i+1)+'.iloc[:___]') for i in range(20)])

# join NSS
df_total = pd.merge(df_total, dfnss, how='inner', left_on=['POL_NUM'], right_on=['scnum'])
df_total.drop('scnum', axis='columns', inplace=True)

#len(df_total) -->  98 605 188
# changing column names to lower 
df_total.columns = df_total.columns.str.lower()
# target  
target = df_total['pv_cf_rdr']
target.head()
# drop targett from total df
df_total.drop(['pv_cf_rdr'], axis = 1, inplace = True)
# std variable cv_ps_0_std
df_total['cv_ps_0_std'] = df_total['cv_ps_0'].apply(lambda x: ( x-mean(df_total['cv_ps_0']))/math.sqrt(st.variance(df_total['cv_ps_0'])))
# count month between two dates
def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month
# delaring variable current date
cd = datetime.datetime(2020, 12, 31)

df_total['inc_date_ct'] = df_total['inc_date'].apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y'))
df_total['cnt_months'] = df_total['inc_date_ct'].apply(lambda x: diff_month(x, cd))
# drop all columns thay are not used
df_total.drop(['inc_date', 'inc_date_ct', 'cv_ps_0'], axis = 1, inplace = True) 



# split train and test
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

SEED = 500
X_train, X_test, y_train, y_test= train_test_split(
    df_total, #explanatory
    target, #response
    test_size=0.2, #hold out size
    random_state=SEED
    )

#save testing and training data for later use (use list as a container)
with open(r"LLcvalue.pickle", "wb") as output_file:
    pickle.dump([X_train, y_train, X_test, y_test], output_file) #dump
#This file will be used in later scripts.

#You can load the file containing variables [X_train, y_train, X_test, y_test]
import pickle
with open(r"LLcvalue.pickle", "rb") as input_file:
    X_train, y_train, X_test, y_test = pickle.load(input_file)


#%reset -f
SEED = 500

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

#instantiate XGBRegressor 
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

=======

import pandas as pd 
import numpy as np
from statistics import mean
import math
from datetime import datetime
import statistics as st
import datetime
import pickle


# load all files and get random intances
RATIO = 0.0001 # portion of dataset

def df_get_instances(df, ratio):
    df = df.iloc[np.random.randint(0, df.shape[0], int(df.shape[0] * ratio))]
    return df

df1 = pd.read_csv('scen_0001-0200.csv')
df1 = df_get_instances(df1, RATIO)

df2 = pd.read_csv('scen_0201-0500.csv')
df2 = df_get_instances(df2, RATIO)

df3 = pd.read_csv('scen_0501-0700.csv')
df3 = df_get_instances(df3, RATIO)

df4 = pd.read_csv('scen_0701-1000.csv')
df4 = df_get_instances(df4, RATIO)

df5 = pd.read_csv('scen_1001-1300.csv')
df5 = df_get_instances(df5, RATIO)

df6 = pd.read_csv('scen_1301-1600.csv')
df6 = df_get_instances(df6, RATIO)

df7 = pd.read_csv('scen_1601-1900.csv')
df7 = df_get_instances(df7, RATIO)

df8 = pd.read_csv('scen_1901-2236.csv')
df8 = df_get_instances(df8, RATIO)

df9 = pd.read_csv('scen_6001-6300.csv')
df9 = df_get_instances(df9, RATIO)

df10 = pd.read_csv('scen_6301-6600.csv')
df10 = df_get_instances(df10, RATIO)

df11 = pd.read_csv('scen_6601-6900.csv')
df11 = df_get_instances(df11, RATIO)

df12 = pd.read_csv('scen_6901-7200.csv')
df12 = df_get_instances(df12, RATIO)

df13 = pd.read_csv('scen_7201-7500.csv')
df13 = df_get_instances(df13, RATIO)

df14 = pd.read_csv('scen_7501-7800.csv')
df14 = df_get_instances(df14, RATIO)

df15 = pd.read_csv('scen_7801-8236.csv')
df15 = df_get_instances(df15, RATIO)

df16 = pd.read_csv('scen_9001-9300.csv')
df16 = df_get_instances(df16, RATIO)

df17 = pd.read_csv('scen_9301-9700.csv')
df17 = df_get_instances(df17, RATIO)

df18 = pd.read_csv('scen_9701-10000.csv')
df18 = df_get_instances(df18, RATIO)

df19 = pd.read_csv('scen_10001-10300.csv')
df19 = df_get_instances(df19, RATIO)

df20 = pd.read_csv('scen_10301-10600.csv')
df20 = df_get_instances(df20, RATIO)

df21 = pd.read_csv('scen_10601-10900.csv')
df21 = df_get_instances(df21, RATIO)

df22 = pd.read_csv('scen_10901-11236.csv')
df22 = df_get_instances(df22, RATIO)

#load curves following NSS optimization
dfnss = pd.read_csv('nss_tau2.csv')
dfnss.drop(['Unnamed: 0'], axis='columns', inplace=True)

# join them together
df_total = pd.concat([eval('df'+str(i+1)) for i in range(22)])

# join them together, ___ rows
#df_total = pd.concat([eval('df'+str(i+1)+'.iloc[:___]') for i in range(20)])

# join NSS
df_total = pd.merge(df_total, dfnss, how='inner', left_on=['POL_NUM'], right_on=['scnum'])
df_total.drop('scnum', axis='columns', inplace=True)

#len(df_total) -->  98 605 188
# changing column names to lower 
df_total.columns = df_total.columns.str.lower()
# target  
target = df_total['pv_cf_rdr']
target.head()
# drop targett from total df
df_total.drop(['pv_cf_rdr'], axis = 1, inplace = True)
# std variable cv_ps_0_std
df_total['cv_ps_0_std'] = df_total['cv_ps_0'].apply(lambda x: ( x-mean(df_total['cv_ps_0']))/math.sqrt(st.variance(df_total['cv_ps_0'])))
# count month between two dates
def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month
# delaring variable current date
cd = datetime.datetime(2020, 12, 31)

df_total['inc_date_ct'] = df_total['inc_date'].apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y'))
df_total['cnt_months'] = df_total['inc_date_ct'].apply(lambda x: diff_month(x, cd))
# drop all columns thay are not used
df_total.drop(['inc_date', 'inc_date_ct', 'cv_ps_0'], axis = 1, inplace = True) 



# split train and test
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

SEED = 500
X_train, X_test, y_train, y_test= train_test_split(
    df_total, #explanatory
    target, #response
    test_size=0.2, #hold out size
    random_state=SEED
    )

#save testing and training data for later use (use list as a container)
with open(r"LLcvalue.pickle", "wb") as output_file:
    pickle.dump([X_train, y_train, X_test, y_test], output_file) #dump
#This file will be used in later scripts.

#You can load the file containing variables [X_train, y_train, X_test, y_test]
import pickle
with open(r"LLcvalue.pickle", "rb") as input_file:
    X_train, y_train, X_test, y_test = pickle.load(input_file)


#%reset -f
SEED = 500

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

#instantiate XGBRegressor 
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

>>>>>>> 03a2d78d628ebd402375fd5e0f59f54a5ac53fba
