import pandas as pd 
import numpy as np
from statistics import mean
import math
from datetime import datetime
import statistics as st
import datetime



# load all files
df1 = pd.read_csv('./dataset/scen_0001-0200.csv')
df2 = pd.read_csv('./dataset/scen_0201-0500.csv')
df3 = pd.read_csv('./dataset/scen_0501-0700.csv')
df4 = pd.read_csv('./dataset/scen_0701-1000.csv')
df5 = pd.read_csv('./dataset/scen_1001-1300.csv')
df6 = pd.read_csv('./dataset/scen_1301-1600.csv')
df7 = pd.read_csv('./dataset/scen_1601-1900.csv')
df8 = pd.read_csv('./dataset/scen_1901-2236.csv')
df9 = pd.read_csv('./dataset/scen_6001-6300.csv')
df10 = pd.read_csv('./dataset/scen_6301-6600.csv')
df11 = pd.read_csv('./dataset/scen_6601-6900.csv')
df12 = pd.read_csv('./dataset/scen_6901-7200.csv')
df13 = pd.read_csv('./dataset/scen_7201-7500.csv')
df14 = pd.read_csv('./dataset/scen_7501-7800.csv')
df15 = pd.read_csv('./dataset/scen_7801-8236.csv')
df16 = pd.read_csv('./dataset/scen_9001-9300.csv')
df17 = pd.read_csv('./dataset/scen_9301-9700.csv')
df18 = pd.read_csv('./dataset/scen_9701-10000.csv')
df19 = pd.read_csv('./dataset/scen_10001-10300.csv')
df20 = pd.read_csv('./dataset/scen_10301-10600.csv')
df21 = pd.read_csv('./dataset/scen_10601-10900.csv')
df22 = pd.read_csv('./dataset/scen_10901-11236.csv')
# join them together
df_total =  pd.concat([eval('df'+str(i+1)) for i in range(22)])
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
df_total.drop([['inc_date', 'inc_date_ct', 'cv_ps_0']], axis = 1, inplace = True) 
