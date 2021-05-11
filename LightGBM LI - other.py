
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

# Evaluate the test set MAPEscore
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

# Evaluate the test set MAPEscore
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

