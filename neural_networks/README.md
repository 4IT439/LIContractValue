# Predicting the PVCF using Neural Networks

Branch of the project aiming to predict PVCF using Neural Networks.

# Content of the directory

## pvcf_nn_train.py

The script used for a training of a sequential model - neural network - with parameters passed as arguments.

**Arguments:**
- `-b` / `--basedir` - directory where file with source data is stored.
  - default='./data/'
- `-f` / `--filename` - name of a file with source data.
  - default='df_merged_train_test.pickle'
- `-a` / `--architecture` - architecture of the NN in a list where elements are number of neurons in the layers.
  - default='[256,256,256,1]'
- `--frac` - a fraction of the original data used for training and validation.
  - default='0.2'
- `--activation` - activation function for hidden layers.
  - default='relu'
- `--optimizer` - optimizer used for the model.
  - default='Adam'
- `-l` / `--loss` - loss function for hidden layers.
  - default='mse'
- `--fit_val_split` - a fraction of the training data used as validation data during the training of the model.
  - default='0.25'
- `--batch` - number of data samples used for one forward and backward pass during the training.
  - default='256'
- `-e` / `--epochs` - a maximum number of epochs of training, in case that early stop wont be executed.
  - default='5'
- `-p` / `--patience` - number of epochs with no improvement after which training will be stopped.
  - default='5'
- `-o` / `--output_file` - prefix of the name of a output file.
  - default='out'
- `--ncpus_inter` - maximum number of cpus allowed to use for Tensorflow globally.
  - default='4'
- `--ncpus_intra` - maximum number of cpus allowed to use for Tensorflow locally (within a single node).
  - default='4'
- `--export_predictions` - whether to export a file with predictions for validation data.
  - default='False'
- `--load_model` - path to the model to load.
  - default=''
- `--initial_epoch` - epoch number from which training should start.
  - default='0'

**Example of running:**
`python ./pvcf_nn_train.py -b ./ -f df_merged_train_test.pickle -a [256,256,256,1] --frac 0.2 --activation relu --optimizer Adam -l mse --fit_val_split 0.25 --batch 256 -e 100 -p 5 -o out --ncpus_inter 4 --ncpus_intra 4 --export_predictions True --export_model True --load_model "" --initial_epoch 0`

This will execute training of the neural network with the data in `pickle` format (`df_merged_train_test.pickle`) stored in the actual directory (`./`). The model will have input layer with 256 neurons (dealing possible difference in a number of the features in the data is secured), 2 hidden layers of 256 neurons each, and output layer of 1 neuron without the activation function (since it is a regression problem). 20% of the original training data will be used for training. `relu` function will be used as an activation function in the input layer and hidden layers. The model will be compiled with `Adam` optimizer and `mse` (mean squared error) as a loss function. `Batch` of 256 data points will be used in one pass during the training. Training will run for 100 `epochs` with `patience` of 5 epochs for early stopping. The `output` with the training and evaluation statistics will be saved in the file with "out" prefix which will be followed with some of the job configuration values and the timestamp of the end of the job, delimited by "_" character. A second output file - one with predictions - will be saved to a file with the same prefix that the previous one and will contain two columns - true and predicted target value for each sample from validation set. A trained model will be exported too. The Tensorflow will be limited with 4 cpus globally with maximum of 4 cpus within a single node.

**Example of output:**

Example name of the output file: `out_2021-09-10T22-33-30`.

A name of the output file consist of the following values:
- given prefix,
- timestamp of the end of the job,
delimited by "_" character.

Example content of the output file:
```
# train samples: 11243987
# val samples:   3747996
# test samples:  797022

Job configuration:
ncpus_inter: 192
ncpus_intra: 32
batch size: 1024

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 15)                240       
_________________________________________________________________
dense_1 (Dense)              (None, 2048)              32768     
_________________________________________________________________
dense_2 (Dense)              (None, 1024)              2098176   
_________________________________________________________________
dense_3 (Dense)              (None, 512)               524800    
_________________________________________________________________
dense_4 (Dense)              (None, 256)               131328    
_________________________________________________________________
dense_5 (Dense)              (None, 128)               32896     
_________________________________________________________________
dense_6 (Dense)              (None, 1)                 129       
=================================================================
Total params: 2,820,337
Trainable params: 2,820,337
Non-trainable params: 0
_________________________________________________________________

History: {'loss': [394385664.0, 14274385.0, 10208155.0, ...], 'val_loss': [34972816.0, 14652837.0, 14832309.0, ...]}
Epochs taken: 100
Train MAPE: 5.936374211777676
Val MAPE:   6.193524478325577
Percentage error for portfolio: -0.025380758471296016%
MAE:  483.2662332849267
MSE:  916312.0112955284
RMSE: 957.2418771112808
Fit time: [min]        462.3952811638514
Eval train time [min]: 23.928626724084218
Eval val time [min]:   7.974666937192281
Global time [min]:     494.6127783497175
```

An output file contains numbers of training, validation and test samples used in the execution, job configuration (given resources), informations about the model (layers and parameters), history of the training (list of values of training and validation loss function during the training), times elapsed while training and evaluating the model and various performance metrics.

**Example of predictions file:**

Example name of the output file: `out_predictions_2021-09-10T22-33-30`.

```
y_true,y_pred
-115821.264904,-119294.56
-67939.443203,-68778.42
1569.292885,-477.37497
-167447.782266,-163292.95
-124257.423452,-125826.59
-308250.246582,-310408.44
9281.726914,10019.204
-103398.421238,-97874.31
-98017.382675,-97448.43
...
-38590.240099,-37963.68
-150685.84767,-152547.66
-154427.704847,-155155.58
-75243.79394,-75227.72
-104535.168481,-105675.02
-78143.959917,-77925.42
10687.23709,11103.448
-780188.750392,-783215.5
-103329.938404,-104489.086
23748.806003,27756.816
```

Prediction file contains two columns. `y_pred` stores true values of target variable (`y`), `y_pred` stores a predicted ones.

**Model output:**

Example name of the output directory: `out_model_2021-09-10T22-33-30`.

This output directory contains a trained tensorflow model. This model can be loaded in another application. 

Example of loading:
```python
from tensorflow import keras
model = keras.models.load_model('path/to/location')
```

## pvcf_nn.ipynb

Jupyter notebook with almost the same content as `pvcf_nn_train.py` intended for interactive work.

## environment.yml

Definition of the Conda environment intended for creating Python virtual env for running the `pvcf_nn_train.py` script.

## pvcfJob.sh

Scheduling script used by job scheduler to run the script.
