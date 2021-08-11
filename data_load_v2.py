#!/usr/bin/env python3
import pandas as pd 
import numpy as np
from statistics import mean
import math
from datetime import datetime
import statistics as st
import datetime
import sklearn 
from sklearn.model_selection import train_test_split

import pickle #load data in binary format

FRAC = 0.1
T_SET_RATIO = 0.05
STAND = False


#load ir 
df_ir = pd.read_csv('nss_tau2.CSV')
# set #sc number as index 
df_ir  = df_ir.set_index('scnum')
df_ir.head()
df_ir.drop('Unnamed: 0', axis = 1, inplace = True)


# load all files
df1 = pd.read_csv('scen_0001-0200.CSV')
df2 = pd.read_csv('scen_0201-0500.CSV')
df3 = pd.read_csv('scen_0501-0700.CSV')
df4 = pd.read_csv('scen_0701-1000.CSV')
df5 = pd.read_csv('scen_1001-1300.CSV')
df6 = pd.read_csv('scen_1301-1600.CSV')
df7 = pd.read_csv('scen_1601-1900.CSV')
df8 = pd.read_csv('scen_1901-2236.CSV')
df9 = pd.read_csv('scen_3001-3300.CSV')
df10 = pd.read_csv('scen_3301-3600.CSV')
df11 = pd.read_csv('scen_3601-3900.CSV')
df12 = pd.read_csv('scen_3901-4200.CSV')
df13 = pd.read_csv('scen_4201-4700.CSV')
df14 = pd.read_csv('scen_4701-5000.CSV')
df15 = pd.read_csv('scen_5001-5236.CSV')
df16 = pd.read_csv('scen_6001-6300.CSV')
df17 = pd.read_csv('scen_6301-6600.CSV')
df18 = pd.read_csv('scen_6601-6900.CSV')
df19 = pd.read_csv('scen_6901-7200.CSV')
df20 = pd.read_csv('scen_7201-7500.CSV')
df21 = pd.read_csv('scen_7501-7800.CSV')
df22 = pd.read_csv('scen_7801-8236.CSV')
df23 = pd.read_csv('scen_9001-9300.CSV')
df24 = pd.read_csv('scen_9301-9700.CSV')
df25 = pd.read_csv('scen_9701-10000.CSV')
df26 = pd.read_csv('scen_10001-10300.CSV')
df27 = pd.read_csv('scen_10301-10600.CSV')
df28 = pd.read_csv('scen_10601-10900.CSV')
df29 = pd.read_csv('scen_10901-11236.CSV')
df30 = pd.read_csv('scen_12001-12300.CSV')
df31 = pd.read_csv('scen_12301-12600.CSV')
df32 = pd.read_csv('scen_12601-12900.CSV')
df33 = pd.read_csv('scen_12901-13200.CSV')
df34 = pd.read_csv('scen_13201-13500.CSV')
df35 = pd.read_csv('scen_13501-13800.CSV')
df36 = pd.read_csv('scen_13801-14236.CSV')

#making sample from them 
df_final = pd.DataFrame()
for i in range(1,37):
    #k = pd.DataFrame()
    df_final  = pd.concat([df_final, eval('df'+str(i)).sample(frac=FRAC, replace = False, random_state = np.random.RandomState())])
# final df has 19 721 038 rows 
df_final.head()
# changing columns from cap to lower
df_final.columns = df_final.columns.str.lower()
# joined with ir scenarios 
df_merged = df_final.merge(df_ir, how = 'left',left_on = 'ir_scen', right_on = 'scnum')
# drop column with the scenario number, is not relevant anymore
df_merged.drop(['ir_scen', 'pol_num'], axis = 1, inplace = True)
target = df_merged['pv_cf_rdr'].copy()

# drop target from total df
df_merged = df_merged.drop(['pv_cf_rdr'], axis = 1)
df_merged.head()
# std variable cv_ps_0_std
if STAND:
    mm = mean(df_merged['cv_ps_0'])
    std  = math.sqrt(st.variance(df_merged['cv_ps_0']))
    df_merged['cv_ps_0_std'] = df_merged['cv_ps_0'].apply(lambda x: (x-mm/std))
    df_merged.head()

# count month between two dates
def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

# delaring variable current date
cd = datetime.datetime(2020, 12, 31)
df_merged.head()
df_merged['inc_date_ct'] = df_merged['inc_date'].apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y'))
df_merged['cnt_months'] = df_merged['inc_date_ct'].apply(lambda x: diff_month(cd, x))

df_merged.head()

# drop all columns thay are not used
df_merged = df_merged.drop((['inc_date', 'inc_date_ct', 'cv_ps_0'] if STAND==True \
    else ['inc_date', 'inc_date_ct']), axis = 1) 
df_merged.head()

SEED = 500
X_train, X_test, y_train, y_test= train_test_split(
    df_merged, #explanatory
    target, #response
    test_size=T_SET_RATIO, #hold out size
    random_state=SEED
    )

y_train.dtypes
#save testing and training data for later use (use list as a container)
with open(r"df_merged_train_test.pickle", "wb") as output_file:
    pickle.dump([X_train, y_train, X_test, y_test], output_file)

