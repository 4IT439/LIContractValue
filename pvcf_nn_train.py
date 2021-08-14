import pandas as pd 
import numpy as np
import time
from datetime import datetime
import argparse
import ast
import sys

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('-b', '--basedir', help='Directory where file with source data is stored.')
parser.add_argument('-f', '--filename', help='Name of a file with source data.')
parser.add_argument('-a', '--architecture', help='Architecture of the NN in a list where elements are number of neurons in the layers.')
parser.add_argument('--frac', help='A fraction of the original data used for training and validation.')
parser.add_argument('--train_size', help='A fraction of the data used for training.')
parser.add_argument('--val_size', help='A fraction of the data used for validation during evaluation.')
parser.add_argument('--activation', help='Activation function for hidden layers.')
parser.add_argument('--optimizer', help='Optimizer used for the model.')
parser.add_argument('-l', '--loss', help='Loss function for hidden layers.')
parser.add_argument('--fit_val_split', help='A fraction of the training data used as validation data during the training of the model.')
parser.add_argument('--batch', help='Number of data samples used for one forward and backward pass during the training.')
parser.add_argument('-e', '--epochs', help='A maximum number of epochs of training, in case that early stop wont be executed.')
parser.add_argument('-p', '--patience', help='Number of epochs with no improvement after which training will be stopped.')
parser.add_argument('-o', '--output_file', help='Name of a output file.')

args=parser.parse_args()

TARGET = 'pv_cf_rdr'
SEED = 420  # fix a seed for randomized functions

BASEDIR = args.basedir  # directory where file with source data is stored
FILENAME = args.filename  # name of a file with source data
OUTPUT_FILE = args.output_file + '_' + datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

FRAC = float(args.frac)  # a fraction of the original data used for training and validation 
TRAIN_SIZE = float(args.train_size)  # a fraction of the data used for training
VAL_SIZE = float(args.val_size)  # a fraction of the data used for validation during evaluation

ARCHITECTURE = ast.literal_eval(args.architecture)
LAYERS = len(ARCHITECTURE)
ACTIVATION = args.activation
OPTIMIZER = args.optimizer
LOSS = args.loss
MODEL_FIT_VAL_SPLIT = float(args.fit_val_split)  # a fraction of the training data used as validation data during the training of the model  
BATCH_SIZE = int(args.batch)  # number of data samples used for one forward and backward pass during the training
EPOCHS_MAX = int(args.epochs)  # a maximum number of epochs of training, in case that early stop wont be executed
ES_PATIENCE = int(args.patience)  # number of epochs with no improvement after which training will be stopped

# load preprocessed data
df = pd.read_pickle(BASEDIR + FILENAME)

X_train = df[0]
y_train = df[1]
X_test = df[2]
y_test = df[3]

# cast suitable features to the smaller type
X_train.frequency = X_train.frequency.astype('int8')
X_train.sum_ins = X_train.sum_ins.astype('int8')
X_train.pol_period = X_train.pol_period.astype('int8')
X_train.sex = X_train.sex.astype('int8')
X_train.entry_age = X_train.entry_age.astype('int8')
X_train.cnt_months = X_train.cnt_months.astype('int8')

X_test.frequency = X_test.frequency.astype('int8')
X_test.sum_ins = X_test.sum_ins.astype('int8')
X_test.pol_period = X_test.pol_period.astype('int8')
X_test.sex = X_test.sex.astype('int8')
X_test.entry_age = X_test.entry_age.astype('int8')
X_test.cnt_months = X_test.cnt_months.astype('int8')

# pick a fraction of the data for training
xy_train = pd.concat([X_train, y_train], axis=1)
xy_train_subset = xy_train.sample(frac=FRAC, replace = False, random_state = np.random.RandomState())

y_train = xy_train_subset[TARGET]
X_train = xy_train_subset.drop(columns=[TARGET])

# pick a validation data
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, train_size=TRAIN_SIZE, test_size=VAL_SIZE, random_state=SEED)

# one hot encoding
dummies_train = pd.get_dummies(X_train['frequency'])
dummies_val = pd.get_dummies(X_val['frequency'])
dummies_test = pd.get_dummies(X_test['frequency'])

dummies_train.columns = ['frequency_1', 'frequency_2', 'frequency_4', 'frequency_11', 'frequency_12']
dummies_val.columns = ['frequency_1', 'frequency_2', 'frequency_4', 'frequency_11', 'frequency_12']
dummies_test.columns = ['frequency_1', 'frequency_2', 'frequency_4', 'frequency_11', 'frequency_12']

X_train = X_train.drop('frequency', axis=1)
X_val = X_val.drop('frequency', axis=1)
X_test = X_test.drop('frequency', axis=1)

X_train = X_train.join(dummies_train)
X_val = X_val.join(dummies_val)
X_test = X_test.join(dummies_test)

# standardize data
X_train_toscale = X_train[['sum_ins', 'pol_period', 'entry_age', 'beta0', 'beta1', 'beta2', 'tau', 'cv_ps_0', 'cnt_months']]
X_val_toscale = X_val[['sum_ins', 'pol_period', 'entry_age', 'beta0', 'beta1', 'beta2', 'tau', 'cv_ps_0', 'cnt_months']]
X_test_toscale = X_test[['sum_ins', 'pol_period', 'entry_age', 'beta0', 'beta1', 'beta2', 'tau', 'cv_ps_0', 'cnt_months']]

X_train_nottoscale = X_train[['sex', 'frequency_1', 'frequency_2', 'frequency_4', 'frequency_11', 'frequency_12']]
X_val_nottoscale = X_val[['sex', 'frequency_1', 'frequency_2', 'frequency_4', 'frequency_11', 'frequency_12']]
X_test_nottoscale = X_test[['sex', 'frequency_1', 'frequency_2', 'frequency_4', 'frequency_11', 'frequency_12']]

standard_scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    standard_scaler.fit_transform(X_train_toscale), columns=X_train_toscale.columns, index=X_train_toscale.index)
X_val_scaled = pd.DataFrame(
    standard_scaler.transform(X_val_toscale), columns=X_val_toscale.columns, index=X_val_toscale.index)
X_test_scaled = pd.DataFrame(
    standard_scaler.transform(X_test_toscale), columns=X_test_toscale.columns, index=X_test_toscale.index)

X_train = pd.concat([X_train_scaled, X_train_nottoscale], axis=1)
X_val = pd.concat([X_val_scaled, X_val_nottoscale], axis=1)
X_test = pd.concat([X_test_scaled, X_test_nottoscale], axis=1)

# define model
model = Sequential()
for layer in range(LAYERS):
    if layer == 0:
        model.add(Dense(ARCHITECTURE[layer], activation=ACTIVATION, input_shape=(15,)))
    elif layer > 0 and layer < LAYERS-1:
        model.add(Dense(ARCHITECTURE[layer], activation=ACTIVATION))
    else:
        model.add(Dense(ARCHITECTURE[layer]))

model.compile(optimizer=OPTIMIZER, loss=LOSS)

# early stopping definition
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=ES_PATIENCE)

# fit model
start = time.time()
history = model.fit(
    verbose=0,
    x=X_train,
    y=y_train,
    validation_split=MODEL_FIT_VAL_SPLIT,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS_MAX,
    initial_epoch=0,
    callbacks=[es]
)
end = time.time()
fit_time = end - start

# evaluate model
start = time.time()
y_pred_train = model.predict(X_train)
end = time.time()
eval_train_time = end - start

start = time.time()
y_pred_val = model.predict(X_val)
end = time.time()
eval_val_time = end - start

with open(OUTPUT_FILE, "w") as of:
    of.write('# train samples: ' + str(len(X_train)) + '\n')
    of.write('# val samples:   ' + str(len(X_val)) + '\n')
    of.write('# test samples:  ' + str(len(X_test)) + '\n')
    of.write('\n')
    
    model.summary(print_fn=lambda x: of.write(x + '\n'))
    of.write('\n')
    
    of.write('History: ' + str(history.history) + '\n')
    of.write('Fit time: [s]        ' + str(fit_time) + '\n')
    of.write('Eval train time [s]: ' + str(eval_train_time) + '\n')
    of.write('Eval val time [s]:   ' + str(eval_val_time) + '\n')
    of.write('Train MAPE: ' + str(metrics.mean_absolute_percentage_error(y_train, y_pred_train)*100) + '\n')
    of.write('Val MAPE:   ' + str(metrics.mean_absolute_percentage_error(y_val, y_pred_val)*100) + '\n')
    of.write('MAE:  ' + str(metrics.mean_absolute_error(y_val, y_pred_val)) + '\n')  
    of.write('MSE:  ' + str(metrics.mean_squared_error(y_val, y_pred_val)) + '\n')  
    of.write('RMSE: ' + str(np.sqrt(metrics.mean_squared_error(y_val, y_pred_val))) + '\n')
