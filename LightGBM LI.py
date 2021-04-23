
import pandas as pd 
import numpy as np
from statistics import mean
import math
from datetime import datetime
import statistics as st
import datetime
import pickle


# load all files
df1 = pd.read_csv('./dataset/scen_0001-0200.csv').iloc[:1000]
df2 = pd.read_csv('./dataset/scen_0201-0500.csv').iloc[:1000]
df3 = pd.read_csv('./dataset/scen_0501-0700.csv').iloc[:1000]
df4 = pd.read_csv('./dataset/scen_0701-1000.csv').iloc[:1000]
df5 = pd.read_csv('./dataset/scen_1001-1300.csv').iloc[:1000]
df6 = pd.read_csv('./dataset/scen_1301-1600.csv').iloc[:1000]
df7 = pd.read_csv('./dataset/scen_1601-1900.csv').iloc[:1000]
df8 = pd.read_csv('./dataset/scen_1901-2236.csv').iloc[:1000]
df9 = pd.read_csv('./dataset/scen_6001-6300.csv').iloc[:1000]
df10 = pd.read_csv('./dataset/scen_6301-6600.csv').iloc[:1000]
df11 = pd.read_csv('./dataset/scen_6601-6900.csv').iloc[:1000]
df12 = pd.read_csv('./dataset/scen_6901-7200.csv').iloc[:1000]
df13 = pd.read_csv('./dataset/scen_7201-7500.csv').iloc[:1000]
df14 = pd.read_csv('./dataset/scen_7501-7800.csv').iloc[:1000]
df15 = pd.read_csv('./dataset/scen_7801-8236.csv').iloc[:1000]
df16 = pd.read_csv('./dataset/scen_9001-9300.csv').iloc[:1000]
df17 = pd.read_csv('./dataset/scen_9301-9700.csv').iloc[:1000]
df18 = pd.read_csv('./dataset/scen_9701-10000.csv').iloc[:1000]
df19 = pd.read_csv('./dataset/scen_10001-10300.csv').iloc[:1000]
df20 = pd.read_csv('./dataset/scen_10301-10600.csv').iloc[:1000]
#df21 = pd.read_csv('./dataset/scen_10601-10900.csv').iloc[:1000]
#df22 = pd.read_csv('./dataset/scen_10901-11236.csv').iloc[:1000]

dfnss = pd.read_csv('./dataset/nss.csv')
dfnss.drop(['Unnamed: 0'], axis='columns', inplace=True)

# join them together, 500 rows
df_total = pd.concat([eval('df'+str(i+1)+'.iloc[:50]') for i in range(20)])

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

SEED = 500
X_train, X_test, y_train, y_test= train_test_split(
    df_total, #explanatory
    target, #response
    test_size=0.2, #hold out size
    random_state=SEED
    )

#save testing and training data for later use (use list as a container)
with open(r"./dataset/LLcvalue.pickle", "wb") as output_file:
    pickle.dump([X_train, y_train, X_test, y_test], output_file) #dump
#This file will be used in later scripts.
#You can load the file containing variables [X_train, y_train, X_test, y_test]
#with open(r"LLcvalue.pickle", "rb") as input_file:
#    X_train, y_train, X_test, y_test = pickle.load(input_file)


os.add_dll_directory('c:/Users/host/anaconda3/Lib/site-packages/lightgbm')
os.add_dll_directory('c:/Users/host/anaconda3')
os.add_dll_directory('c:/Users/host/anaconda3/lib')

import lightgbm as lgb #pip3 install lightgbm

