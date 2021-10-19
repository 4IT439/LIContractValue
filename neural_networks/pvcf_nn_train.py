import pandas as pd
import numpy as np
import time
from datetime import datetime
import argparse
import ast
import sys

from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

global_start = time.time()
global_end = None

# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('-b', '--basedir', help='Directory where file with source data is stored.', default='./')
parser.add_argument('-f', '--filename', help='Name of a file with source data.', default='df_merged_train_test.pickle')
parser.add_argument('-a', '--architecture', help='Architecture of the NN in a list where elements are number of neurons in the layers.', default='[256,256,256,1]')
parser.add_argument('--frac', help='A fraction of the original data used for training and validation.', default='0.2')
parser.add_argument('--activation', help='Activation function for hidden layers.', default='relu')
parser.add_argument('--optimizer', help='Optimizer used for the model.', default='Adam')
parser.add_argument('-l', '--loss', help='Loss function for hidden layers.', default='mse')
parser.add_argument('--fit_val_split', help='A fraction of the training data used as validation data during the training of the model.', default='0.25')
parser.add_argument('--batch', help='Number of data samples used for one forward and backward pass during the training.', default='256')
parser.add_argument('-e', '--epochs', help='A maximum number of epochs of training, in case that early stop wont be executed.', default='3')
parser.add_argument('-p', '--patience', help='Number of epochs with no improvement after which training will be stopped.', default='5')
parser.add_argument('-o', '--output_file', help='Name of a output file.', default='out')
parser.add_argument('--ncpus_inter', help='Maximum number of cpus allowed to use for Tensorflow globally.', default='4')
parser.add_argument('--ncpus_intra', help='Maximum number of cpus allowed to use for Tensorflow locally (within a single node).', default='4')
parser.add_argument('--export_predictions', help='Whether to export a file with predictions for validation data.', default='False')
parser.add_argument('--export_model', help='Whether to export a trained model.', default='False')
parser.add_argument('--load_model', help='Path to the model to load.', default='')
parser.add_argument('--initial_epoch', help='Epoch number from which training should start.', default='0')

args=parser.parse_args()

# configure tensorflow resource use
tf.config.threading.set_inter_op_parallelism_threads(int(args.ncpus_inter))
tf.config.threading.set_intra_op_parallelism_threads(int(args.ncpus_intra))

TARGET = 'pv_cf_rdr'
SEED = 420  # fix a seed for randomized functions

BASEDIR = args.basedir  # directory where file with source data is stored
FILENAME = args.filename  # name of a file with source data

FRAC = float(args.frac)  # a fraction of the original data used for training and validation 

ARCHITECTURE = ast.literal_eval(args.architecture)
LAYERS = len(ARCHITECTURE)
ACTIVATION = args.activation
OPTIMIZER = args.optimizer
LOSS = args.loss
MODEL_FIT_VAL_SPLIT = float(args.fit_val_split)  # a fraction of the training data used as validation data during the training of the model  
BATCH_SIZE = int(args.batch)  # number of data samples used for one forward and backward pass during the training
EPOCHS_MAX = int(args.epochs)  # a maximum number of epochs of training, in case that early stop wont be executed
ES_PATIENCE = int(args.patience)  # number of epochs with no improvement after which training will be stopped
EXPORT_PREDICTIONS = args.export_predictions
EXPORT_MODEL = args.export_model
LOAD_MODEL = args.load_model
INITIAL_EPOCH = int(args.initial_epoch)

# load preprocessed data
df = pd.read_pickle(BASEDIR + FILENAME)

X_train = df[0]
y_train = df[1]
X_val = df[2]
y_val = df[3]

# cast suitable features to the smaller type
X_train.frequency = X_train.frequency.astype('int8')
X_train.sum_ins = X_train.sum_ins.astype('int8')
X_train.pol_period = X_train.pol_period.astype('int8')
X_train.sex = X_train.sex.astype('int8')
X_train.entry_age = X_train.entry_age.astype('int8')
X_train.cnt_months = X_train.cnt_months.astype('int8')

X_val.frequency = X_val.frequency.astype('int8')
X_val.sum_ins = X_val.sum_ins.astype('int8')
X_val.pol_period = X_val.pol_period.astype('int8')
X_val.sex = X_val.sex.astype('int8')
X_val.entry_age = X_val.entry_age.astype('int8')
X_val.cnt_months = X_val.cnt_months.astype('int8')

# pick a fraction of the data for training
xy_train = pd.concat([X_train, y_train], axis=1)
xy_train_subset = xy_train.sample(frac=FRAC, replace = False, random_state = np.random.RandomState())

y_train = xy_train_subset[TARGET]
X_train = xy_train_subset.drop(columns=[TARGET])

# one hot encoding
dummies_train = pd.get_dummies(X_train['frequency'])
dummies_val = pd.get_dummies(X_val['frequency'])

dummies_train.columns = ['frequency_1', 'frequency_2', 'frequency_4', 'frequency_11', 'frequency_12']
dummies_val.columns = ['frequency_1', 'frequency_2', 'frequency_4', 'frequency_11', 'frequency_12']

X_train = X_train.drop('frequency', axis=1)
X_val = X_val.drop('frequency', axis=1)

X_train = X_train.join(dummies_train)
X_val = X_val.join(dummies_val)

# standardize data
X_train_toscale = X_train[['sum_ins', 'pol_period', 'entry_age', 'beta0', 'beta1', 'beta2', 'tau', 'cv_ps_0', 'cnt_months']]
X_val_toscale = X_val[['sum_ins', 'pol_period', 'entry_age', 'beta0', 'beta1', 'beta2', 'tau', 'cv_ps_0', 'cnt_months']]

X_train_nottoscale = X_train[['sex', 'frequency_1', 'frequency_2', 'frequency_4', 'frequency_11', 'frequency_12']]
X_val_nottoscale = X_val[['sex', 'frequency_1', 'frequency_2', 'frequency_4', 'frequency_11', 'frequency_12']]

standard_scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    standard_scaler.fit_transform(X_train_toscale), columns=X_train_toscale.columns, index=X_train_toscale.index)
X_val_scaled = pd.DataFrame(
    standard_scaler.transform(X_val_toscale), columns=X_val_toscale.columns, index=X_val_toscale.index)

X_train = pd.concat([X_train_scaled, X_train_nottoscale], axis=1)
X_val = pd.concat([X_val_scaled, X_val_nottoscale], axis=1)

# define model
if LOAD_MODEL == '':
    model = Sequential()
    for layer in range(LAYERS):
        if layer == 0:
            model.add(Dense(ARCHITECTURE[layer], activation=ACTIVATION, input_shape=(15,)))
        elif layer > 0 and layer < LAYERS-1:
            model.add(Dense(ARCHITECTURE[layer], activation=ACTIVATION))
        else:
            model.add(Dense(ARCHITECTURE[layer]))
    model.compile(optimizer=OPTIMIZER, loss=LOSS)
elif LOAD_MODEL != '':
    model = tf.keras.models.load_model(LOAD_MODEL)

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
    initial_epoch=INITIAL_EPOCH,
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

timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

OUTPUT_FILE = args.output_file  + '_' + timestamp
    
# calculate percentage error for the whole portfolio
df_predictions = pd.DataFrame()
y_true_series = pd.DataFrame(y_val)
y_pred_series = pd.DataFrame([item for sublist in y_pred_val for item in sublist])

y_true_series.reset_index(drop=True, inplace=True)
y_pred_series.reset_index(drop=True, inplace=True)
df_predictions['y_true'] = y_true_series
df_predictions['y_pred'] = y_pred_series

percentage_error = (df_predictions['y_true'].sum() - df_predictions['y_pred'].sum())/(df_predictions['y_true'].sum())*100

# export file with predictions
if EXPORT_PREDICTIONS == 'True':
    OUTPUT_PREDICTIONS = args.output_file + '_predictions_' + timestamp
    df_predictions.to_csv(OUTPUT_PREDICTIONS, index=False)
  
if EXPORT_MODEL == 'True':
    OUTPUT_MODEL = args.output_file + '_model_' + timestamp
    model.save(OUTPUT_MODEL)

with open(OUTPUT_FILE, "w") as of:
    of.write('# train samples: ' + str(len(X_train)) + '\n')
    of.write('# val samples:   ' + str(len(X_val)) + '\n')
    of.write('\n')
    
    of.write('Job configuration:' + '\n')
    of.write('ncpus_inter: ' + args.ncpus_inter + '\n')
    of.write('ncpus_intra: ' + args.ncpus_intra + '\n')
    of.write('batch size: ' + args.batch + '\n')
    of.write('\n')
    
    model.summary(print_fn=lambda x: of.write(x + '\n'))
    of.write('\n')
    
    of.write('History: ' + str(history.history) + '\n')
    of.write('Epochs taken: ' + str(len(history.history['loss'])) + ' (initial epoch was: ' + str(INITIAL_EPOCH) + ')\n')
    of.write('Train MAPE: ' + str(metrics.mean_absolute_percentage_error(y_train, y_pred_train)*100) + '\n')
    of.write('Val MAPE:   ' + str(metrics.mean_absolute_percentage_error(y_val, y_pred_val)*100) + '\n')
    of.write('Percentage error for portfolio: ' + str(percentage_error) + '%\n')
    of.write('MAE:  ' + str(metrics.mean_absolute_error(y_val, y_pred_val)) + '\n')  
    of.write('MSE:  ' + str(metrics.mean_squared_error(y_val, y_pred_val)) + '\n')  
    of.write('RMSE: ' + str(np.sqrt(metrics.mean_squared_error(y_val, y_pred_val))) + '\n')
    of.write('Fit time: [min]        ' + str(fit_time/60) + '\n')
    of.write('Eval train time [min]: ' + str(eval_train_time/60) + '\n')
    of.write('Eval val time [min]:   ' + str(eval_val_time/60) + '\n')
    global_end = time.time()
    global_time = global_end - global_start
    of.write('Global time [min]:     ' + str(global_time/60) + '\n')
