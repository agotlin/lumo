
# coding: utf-8

# # Lumo Run - Deep FFCN and CNN

# #### Where is this script being run

# In[1]:


machine_to_run_script = 'local' # 'Sherlock', 'local'


# # Load dependencies

# In[2]:


print('Script is starting!') # these are used for Sherlock to check what is causing the hiccup

import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, Input
from keras.layers.normalization import BatchNormalization 
from keras.layers.merge import concatenate
from keras.utils import plot_model
from keras import regularizers 
from keras import optimizers

print('Successfully loaded keras!')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import tensorflow as tf

from datetime import datetime
from dateutil import tz
from IPython import embed
import time
from time import strftime, gmtime
import socket
import pickle
import os.path
#import dill #can't find dill

from pathlib import Path
from sklearn.metrics import confusion_matrix 

np.random.seed(7) # Set seed for reproducibility


# #### Set Hyperparameters

# In[18]:


# Data Setup

num_channels = 6 # number of time-series channels of data (i.e. 7 kinematic features) #NOTE: Change to 6 by removing Pelvic Tilt (recommended by Lumo)
num_anthropometrics = 4 # number of user anthropometric data elements
input_window_size = 26 # number of timestamps used in the input for prediction (i.e. the input window)
label_window_size = 20 # number of timestamps used to label the speed we will be predicting
speed_bucket_size = '0.1' # how to round the data for classification task. Consider '0.5', '0.1', and 'none_use_regression'

previous_model_weights_to_load = "" # If non-empty, load weights from a previous model (note: architectures must be identical)
model_architecture = 'CNN' # 'FCN', 'CNN'
data_input_table_structure = 'Raw_Timeseries' # 'Vectorized_By_Row' 'Raw_Timeseries'
if machine_to_run_script == 'local':
    folder_head_loc = '../';
    folder_data_loc = 'C:/Users/adam/Documents/Lumo/Lumo Data/'
elif machine_to_run_script == 'Sherlock':
    folder_head_loc = '/home/users/agotlin/lumo/'
    folder_data_loc = '/home/users/agotlin/SherlockDataFiles/'
myFileName = 'TimeSeries_InputRaw_1000Runs'
myFileLocation = folder_data_loc + myFileName + '.csv'
    # Other data files/folders to potentially use:
    # 'TimeSeries_InputVector_100runs'   |   'TimeSeries_InputVector_15runs'
    # 'TimeSeries_InputRaw_1000Runs'  |  'TimeSeries_InputRaw_1000Runs_QuarterSample'  |  'TimeSeries_InputRaw_1000Runs_Top10kRowsSample'

# Training strategy

batch_size = 50 # we used 50 for CNN, 128 for FCN
learning_rate = 0.0001 # we used 0.001 for FCN, 0.0001 for CNN
training_epochs = 500
optimizer_type = 'gradient' # options are: "adam" , "rmsprop", "gradient" # adam for FCN, gradient for CNN
loss_function = 'categorical_crossentropy' # Other options (from keras defaults or custom) include: 'categorical_crossentropy' ,'mse', 'mae', 'class_mse', 'class_mae'    
    
# Fully Connected Architecture

num_hidden_units_fc_layers = [256, 256, 256, 128, 128, 128]
hidden_units_strategy = ''.join(str(num) + "_" for num in num_hidden_units_fc_layers) # document strategy 
num_hidden_fc_layers = len(num_hidden_units_fc_layers) # document strategy
activations_fc_layers = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu']
activations_strategy = ''.join(str(num) + "_" for num in activations_fc_layers) # document strategy
dropout_rate_fc_layers = [1.0, 1.0, 1.0, 0.8, 0.8, 0.8]
dropout_rates = ''.join(str(num) + "_" for num in dropout_rate_fc_layers) # document strategy

# Convolutional Architecture
    
sample_stride = input_window_size/2 # how many timestamps to shift over between each unique training example # 18, input_window_size/2
num_filters = 40 # number of filters in Conv2D layer (aka depth) # we used 40, ex; used 128
kernel_size = 5 # kernal size of the Conv2D layer # we use 6, example used 2, I would guess closer to 3
activation_conv_layer = "relu" # options are "relu" , "tanh" and "sigmoid" - used for depthwise_conv
max_pool_kernel_size = 5 # max pooling window size# we use 6, example used 2, I don't agre with 6
conv_layer_dropout = 0.2 # dropout ratio for dropout layer # we don't use in our model

num_hidden_units_fc_layers_CNN = [50]
hidden_units_strategy_CNN = ''.join(str(num) + "_" for num in num_hidden_units_fc_layers_CNN) # document strategy 
num_hidden_fc_layers_CNN = len(num_hidden_units_fc_layers_CNN) # document strategy
activations_fc_layers_CNN = ['tanh']
activations_strategy_CNN = ''.join(str(num) + "_" for num in activations_fc_layers_CNN) # document strategy
dropout_rate_fc_layers_CNN = [1.0]
dropout_rates_CNN = ''.join(str(num) + "_" for num in dropout_rate_fc_layers_CNN) # document strategy


# #### Set Up Automatic Reporting and Plotting

# In[19]:


# Choose the 3 most interesting evaluation metrics to report on in final plots

accuracy_reporting_metric_1 = 'class_mae' # options: 'acc', 'class_percent_1buckRange', 'class_percent_2buckRange'
dev_reporting_metric_1 = 'val_' + accuracy_reporting_metric_1
accuracy_reporting_metric_2 = 'class_percent_2buckRange' # options: 'acc', 'class_percent_1buckRange', 'class_percent_2buckRange'
dev_reporting_metric_2 = 'val_' + accuracy_reporting_metric_2
accuracy_reporting_metric_3 = 'class_mse' # options: s'acc', 'class_percent_1buckRange', 'class_percent_2buckRange'
dev_reporting_metric_3 = 'val_' + accuracy_reporting_metric_3

plt.style.use('ggplot') # style of matlab plots to produce


# In[20]:


# File naming conventions

file_name = strftime("%Y%m%d_%H%M%S", gmtime()) # user input for filename of saved model
plot_note = ""
model_to_lod = ""
results_file_name = "Default_Model_Results_Table_20180726" + "_" + model_architecture

customize_file_names = False
if customize_file_names:
    file_name = input("String to add to model filename (defaults to time stamp if nothing entered):")  
    results_file_name = input("Name of the results file, a table, to store the prediction results") # name of results file
    plot_note = input("Note you'd like to add in the legend of the primary learning curves plot:") #user input to add note to plot
    model_to_load = input("Enter the model name to load to initialize parameters - leave blank to start fresh") #user input to load prev model


# #### Define functions for data processing and plotting

# In[21]:


def read_data(file_path):
    data = pd.read_csv(file_path,header = 0) # This uses the header row (row 0) as the column names
    return data

def windows(data, size): # define time windows to create each training example
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += sample_stride # other common options: (size / 2)
    
# Used for vectorized input WITH SQL pre-processing
def segment_signal_FCN_vector(data_inputs, data_full): 
    #dataframe_input = data_inputs.loc[:, 'gender':'pelvic_tilt_lag_0'] # select all columns from gender to pelvic_tilt_lag_0
    dataframe_input = data_inputs.loc[:, 'gender':'pelvic_rotation_lag_0'] # select all columns from gender to pelvic_rotation_lag_0
    dataframe_labels = data_full.loc[:, 'gps_speed_lag_7':'gps_speed_lag_3'] # select all columns from gender to pelvic_tilt_lag_0
    segments = dataframe_input.values
    labels_before_avg = dataframe_labels.values
    if speed_bucket_size == '0.1':
        labels = np.around(np.mean(labels_before_avg, axis=1),decimals=1)
    elif speed_bucket_size == 'none_use_regression':
        labels = np.mean(labels_before_avg, axis=1)
    return segments, labels

# Used for CNN/FFCN input WITHOUT SQL pre-processing
def segment_signal_w_concat(data_inputs, data_full, input_window_size = input_window_size):
    # define segment shape for training example input
    if  model_architecture == 'FCN':
        segments = np.empty((0,input_window_size*num_channels + num_anthropometrics))
    elif model_architecture == 'CNN':
        segments = np.empty((0,input_window_size*num_channels + num_anthropometrics))    
        segments_timeseries = np.empty((0, input_window_size, num_channels))
        segments_anthro = np.empty((0, num_anthropometrics))
    labels = np.empty((0))
    for (start, end) in windows(data_full['timestamp'], input_window_size):
        a = data_inputs["bounce"][start:end]
        b = data_inputs["braking"][start:end]
        c = data_inputs["cadence"][start:end]
        d = data_inputs["ground_contact"][start:end]
        e = data_inputs["pelvic_drop"][start:end]
        f = data_inputs["pelvic_rotation"][start:end]
        aa = data_inputs["age"][start]
        bb = data_inputs["weight"][start]
        cc = data_inputs["height"][start]
        dd = data_inputs["gender"][start]   
        if(end < data_full.shape[0] and len(data_full['timestamp'][start:end]) == input_window_size and data_full['activity_id'][start]==data_full['activity_id'][end]):
            # Create segment input arrays
            if  model_architecture == 'FCN':
                segments_toadd = np.vstack([np.dstack([a,b,c,d,e,f])])
                segments_toadd_reshape = segments_toadd.reshape(input_window_size * num_channels)
                segments = np.vstack([segments,np.hstack([aa,bb,cc,dd,segments_toadd_reshape])])
            elif model_architecture == 'CNN':        
                segments_timeseries = np.vstack([segments_timeseries,np.dstack([a,b,c,d,e,f])])
                segments_anthro = np.vstack([segments_anthro,np.hstack([aa,bb,cc,dd])])
            # Create labels array
            start_labeling = np.int(np.floor(start+(end-start)/2) - np.floor(label_window_size/2))
            end_labeling = start_labeling + label_window_size
            if speed_bucket_size == '0.1':
                labels = np.append(labels,np.around(np.mean(data_full["gps_speed_true"][start_labeling:end_labeling]),decimals=1)) # round to nearest decimal
            elif speed_bucket_size == 'none_use_regression':
                labels = np.append(labels,np.mean(data_full["gps_speed_true"][start_labeling:end_labeling])) # no rounding, use regression
    if  model_architecture == 'FCN':
        return segments, labels
    elif model_architecture == 'CNN':            
        return segments_timeseries, segments_anthro, labels

def load_results_file_FCN(results_file_name):
    my_file = Path(folder_head_loc + "Model Performance Tables/" + results_file_name + ".csv")
    if my_file.is_file():
        print("Found results file")
        prev_results=pd.read_csv(my_file,header=0)
        print(list(prev_results.columns.values))
        return prev_results
    else:
        print("no results file found - creating file")
        a=[[model_architecture,
            file_name,
            "na",
            myFileLocation,
            training_epochs,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            batch_size,
            learning_rate,
            speed_bucket_size,
            loss_function,
            input_window_size,
            label_window_size,
            optimizer_type,
            "",
            "",
            "",
            num_hidden_fc_layers,
            hidden_units_strategy,
            activations_strategy,
            dropout_rates
            ]]
        
        df=pd.DataFrame(a, columns=["model type",
                                    "model filename",
                                    "plot filename",
                                    "data filename",
                                    "epochs",
                                    "runtime",
                                    "dev accuracy 1",
                                    "train accuracy 1",
                                    "dev accuracy 2",
                                    "train accuracy 2",
                                    "dev accuracy 3",
                                    "train accuracy 4",
                                    "batch_size",
                                    "learning_rate",
                                    "speed_bucket_size",
                                    "loss_function",
                                    "input_window_size",
                                    "label_window_size",
                                    "optimizer_type",
                                    "evaluation_metric_1",
                                    "evaluation_metric_2",
                                    "evaluation_metric_3",
                                    "num_hidden_fc_layers",
                                    "hidden_units_strategy",
                                    "activations_strategy",
                                    "dropout_rates"])
        
        df.to_csv(folder_head_loc + "Model Performance Tables/" + results_file_name + ".csv",index=False ) 
        return df

def load_results_file_CNN(results_file_name):
    my_file = Path(folder_head_loc + "Model Performance Tables/" + results_file_name + ".csv")
    if my_file.is_file():
        #print("Found results file")
        prev_results=pd.read_csv(my_file,header=0)
        #print(list(prev_results.columns.values))
        return prev_results
    else:
        print("no results file found - creating file")
        a=[[model_architecture,
            file_name,
            "na",
            myFileLocation,
            training_epochs,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            batch_size,
            learning_rate,
            speed_bucket_size,
            loss_function,
            input_window_size,
            label_window_size,
            optimizer_type,
            "",
            "",
            "",
            hidden_units_strategy_CNN,
            num_filters,
            kernel_size,
            sample_stride,
            activation_conv_layer,
            activations_strategy_CNN,
            max_pool_kernel_size
            ]]
        
        df=pd.DataFrame(a, columns=["model type",
                                    "model filename",
                                    "plot filename",
                                    "data filename",
                                    "epochs",
                                    "runtime",
                                    "dev accuracy 1",
                                    "train accuracy 1",
                                    "dev accuracy 2",
                                    "train accuracy 2",
                                    "dev accuracy 3",
                                    "train accuracy 2",
                                    "batch_size",
                                    "learning_rate",
                                    "speed_bucket_size",
                                    "loss_function",
                                    "input_window_size",
                                    "label_window_size",
                                    "optimizer_type",
                                    "evaluation_metric_1",
                                    "evaluation_metric_2",
                                    "evaluation_metric_3",
                                    "hidden_units_strategy_CNN",
                                    "num_filters",
                                    "kernel_size",
                                    "sample_stride",
                                    "activation_conv_layer",
                                    "activations_strategy_CNN",
                                    "max_pool_kernel_size"])
        
        df.to_csv(folder_head_loc + "Model Performance Tables/" + results_file_name + ".csv",index=False ) 
        return df        


# #### Normalize Data

# In[22]:


dataset = read_data(myFileLocation)

if data_input_table_structure == 'Raw_Timeseries':
    dataset_inputs = dataset.loc[:, 'gender':'pelvic_rotation'] # normalize all columns from gender to pelvic_tilt
    dataset_inputs_normalized = (dataset_inputs - dataset_inputs.mean())/dataset_inputs.std()
elif data_input_table_structure == 'Vectorized_By_Row':
    dataset_inputs = dataset.loc[:, 'gender':'pelvic_rotation_lag_0'] # normalize all columns from gender to pelvic_rotation_lag_0
    dataset_inputs_normalized = (dataset_inputs - dataset_inputs.mean())/dataset_inputs.std()


# In[23]:


print('Successfuly normalized data!')


# #### Preprocess data to input into model

# In[24]:


np_array_file_string_segment = folder_data_loc + "SavedNPArrays/" + str(myFileName) + "_" + model_architecture + "_" + str(input_window_size) + "_" + str(label_window_size) + "_" + str(sample_stride) + "_segment.npy"
np_array_file_string_segment_timeseries = folder_data_loc + "SavedNPArrays/" + str(myFileName) + "_" + model_architecture + "_" + str(input_window_size) + "_" + str(label_window_size) + "_" + str(sample_stride) + "_segment_timeseries.npy"
np_array_file_string_segment_anthro = folder_data_loc + "SavedNPArrays/" + str(myFileName) + "_" + model_architecture + "_" + str(input_window_size) + "_" + str(label_window_size) + "_" + str(sample_stride) + "_segment_anthro.npy"
np_array_file_string_label = folder_data_loc + "SavedNPArrays/" + str(myFileName) + "_" + model_architecture + "_" +  str(input_window_size) + "_" + str(label_window_size) + "_" + str(sample_stride) + "_label.npy"
np_array_file_string_label2num = folder_data_loc + "SavedNPArrays/" + str(myFileName) + "_" + model_architecture + "_" + str(input_window_size) + "_" + str(label_window_size) + "_" + str(sample_stride) + "_label2num.npy"
    
if os.path.isfile(np_array_file_string_label):   # if this file already exists, load the relevant SavedNPArrays
    print('Pulling down ' + np_array_file_string_label)
    if  model_architecture == 'FCN':
        segments = np.load(np_array_file_string_segment, allow_pickle=True)
    elif model_architecture == 'CNN':
        segments_timeseries = np.load(np_array_file_string_segment_timeseries, allow_pickle=True)
        segments_anthro = np.load(np_array_file_string_segment_anthro, allow_pickle=True)
    labels = np.load(np_array_file_string_label, allow_pickle=True)
    labels_to_number = np.load(np_array_file_string_label2num, allow_pickle=True)
else:    # if this file does not exist, run segment_signal method and create np arrays for future use
    print('Creating ' + np_array_file_string_label)
    if data_input_table_structure == 'Raw_Timeseries':
        if  model_architecture == 'FCN':
            segments, labels = segment_signal_w_concat(dataset_inputs_normalized, dataset)
        elif model_architecture == 'CNN':
            segments_timeseries, segments_anthro, labels = segment_signal_w_concat(dataset_inputs_normalized, dataset)
    elif data_input_table_structure == 'Vectorized_By_Row':
        segments, labels = segment_signal_FCN_vector(dataset_inputs_normalized, dataset)
    if speed_bucket_size != 'none_use_regression': # if not using regression, convert to one-hot vector labels
         labels_to_number = np.unique(labels) # Caches "labels_to_number" in order to use in rmse calculation for classification
         labels = np.asarray(pd.get_dummies(labels), dtype = np.int8) # one-hot labels to classify nearest bucket
    if  model_architecture == 'FCN':
        np.save(np_array_file_string_segment, segments, allow_pickle=True)
    elif model_architecture == 'CNN':
        np.save(np_array_file_string_segment_timeseries, segments_timeseries, allow_pickle=True)
        np.save(np_array_file_string_segment_anthro, segments_anthro, allow_pickle=True)
    np.save(np_array_file_string_label, labels, allow_pickle=True)
    np.save(np_array_file_string_label2num, labels_to_number, allow_pickle=True)

num_buckets_total = len(labels[1]) # total number of classification buckets that exist in the dataset (here, classification bucket == classification class)


# In[25]:


print('Successfully preprocessed data!')


# #### Shuffle data into training and dev

# In[26]:


train_dev_split = np.random.rand(len(labels)) < 0.90 # split data into 90% train, 10% dev, based on lenghto of labels

if  model_architecture == 'FCN':
    X_train = segments[train_dev_split]
    X_test = segments[~train_dev_split]
elif model_architecture == 'CNN':
    X_train_timeseries = segments_timeseries[train_dev_split]
    X_test_timeseries = segments_timeseries[~train_dev_split]
    X_train_anthro = segments_anthro[train_dev_split]
    X_test_anthro = segments_anthro[~train_dev_split]

y_train = labels[train_dev_split]
y_test = labels[~train_dev_split]


# #### Implement NN architecture in a Keras model

# In[27]:


def fcnModel():
    model = Sequential()
    # First layer
    model.add(Dense(num_hidden_units_fc_layers[0], activation=activations_fc_layers[0], input_shape=(input_window_size*num_channels + num_anthropometrics,)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate_fc_layers[0]))
    # Intermediate layers
    for L in range(1, num_hidden_fc_layers):
        model.add(Dense(num_hidden_units_fc_layers[L], activation=activations_fc_layers[L]))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate_fc_layers[L]))
    # Last hidden layer
    if speed_bucket_size != 'none_use_regression': # if classification, use softmax for last layer
        model.add(Dense(num_buckets_total, activation='softmax'))
    else:                                          # if regression, use linear for last layer
        model.add(Dense(1,activation='linear'))
    return model

def cnnModel_multInput(): # (inputs, outputs):
    # CNN over time-series data
    input_cnn = Input(shape=(input_window_size, num_channels))
    conv1 = Conv1D(num_filters, kernel_size,activation=activation_conv_layer)(input_cnn)
    pool1 = MaxPooling1D(pool_size=max_pool_kernel_size, padding='valid', strides=(2))(conv1)
    conv2 = Conv1D(num_filters//10, kernel_size, activation=activation_conv_layer)(pool1) # add additional CNN layer
    flat1 = Flatten()(conv2)
    # Include anthropometric data
    input_anthro = Input(shape=(num_anthropometrics,))
    # Concatenate result of CNN with antropometric data
    merged = concatenate([flat1, input_anthro])
    # Add fully connected hident layers after concatenating (at least one)
    fc1 = Dense(num_hidden_units_fc_layers_CNN[0], activation=activations_fc_layers_CNN[0])(merged) # add first fully connected layer
    for L in range(1, num_hidden_fc_layers_CNN):
        # NEED TO CORRECT BEFORE USE    
        #something like this
        fc_L = Dense(num_hidden_units_fc_layers_CNN[L], activation=activations_fc_layers_CNN[L])(fc_L-1) # add first fully connected layer
        #model.add(BatchNormalization())
        #model.add(Dropout(dropout_rate_fc_layers_CNN[L]))
    if speed_bucket_size != 'none_use_regression': # if classification, use softmax for last layer
        output = Dense(num_buckets_total, activation='softmax')(fc1) # will need to change with more fc layers
    else:                                          # if regression, use linear for last layer
        output = Dense(1,activation='linear')(fc1)  
    model = Model(inputs = [input_cnn, input_anthro], outputs = output)
    return model


# In[28]:


if  model_architecture == 'FCN':
    model = fcnModel()
elif model_architecture == 'CNN':
    model = cnnModel_multInput()


# In[29]:


# View model summary
model.summary()


# #### Define custom loss functions and evaluation metrics

# In[30]:


from keras import backend as K

def class_mse(y_true, y_pred):
    return K.mean(K.square(K.sum(y_pred * labels_to_number,axis=-1,keepdims=True) - K.sum(y_true * labels_to_number,axis=-1,keepdims=True)), axis=-1)
    # Note: we cannot define RMSE directly in Keras since the loss function is defined for one training example at a time

def class_mae(y_true, y_pred):
    return K.mean(K.abs(K.sum(y_pred * labels_to_number,axis=-1,keepdims=True) - K.sum(y_true * labels_to_number,axis=-1,keepdims=True)), axis=-1)

def class_mape(y_true, y_pred):
    diff = K.abs((K.sum(y_true * labels_to_number,axis=-1,keepdims=True) - K.sum(y_pred * labels_to_number,axis=-1,keepdims=True)) / K.clip(K.abs(K.sum(y_true * labels_to_number,axis=-1,keepdims=True)),K.epsilon(),None))
    return 100. * K.mean(diff, axis=-1)

def class_percent_1buckLow(y_true, y_pred): # percent of times the prediction is 1 bucket below the true value
    return K.cast(K.equal(K.cast(K.argmax(y_true, axis=-1), K.floatx()), K.cast(K.argmax(y_pred, axis=-1),K.floatx())+1.0), K.floatx())

def class_percent_2buckLow(y_true, y_pred): # percent of times the prediction is 2 buckets below the true value
    return K.cast(K.equal(K.cast(K.argmax(y_true, axis=-1), K.floatx()), K.cast(K.argmax(y_pred, axis=-1),K.floatx())+2.0), K.floatx())
    
def class_percent_1buckHigh(y_true, y_pred): # percent of times the prediction is 1 bucket above the true value
    return K.cast(K.equal(K.cast(K.argmax(y_true, axis=-1), K.floatx()), K.cast(K.argmax(y_pred, axis=-1),K.floatx())-1.0), K.floatx())    

def class_percent_2buckHigh(y_true, y_pred): # percent of times the prediction is 2 buckets above the true value
    return K.cast(K.equal(K.cast(K.argmax(y_true, axis=-1), K.floatx()), K.cast(K.argmax(y_pred, axis=-1),K.floatx())-2.0), K.floatx())    

def class_percent_1buckRange(y_true, y_pred): # percent of times the prediction is within 1 bucket of true value
    return K.cast(K.equal(K.cast(K.argmax(y_true, axis=-1), K.floatx()), K.cast(K.argmax(y_pred, axis=-1),K.floatx())-1.0), K.floatx()) +     K.cast(K.equal(K.cast(K.argmax(y_true, axis=-1), K.floatx()), K.cast(K.argmax(y_pred, axis=-1),K.floatx())+1.0), K.floatx()) +     K.cast(K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred, axis=-1)),K.floatx())

def class_percent_2buckRange(y_true, y_pred): # percent of times the prediction is within 2 buckets of true value
    return K.cast(K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred, axis=-1)),K.floatx()) +     K.cast(K.equal(K.cast(K.argmax(y_true, axis=-1), K.floatx()), K.cast(K.argmax(y_pred, axis=-1),K.floatx())-1.0), K.floatx()) +     K.cast(K.equal(K.cast(K.argmax(y_true, axis=-1), K.floatx()), K.cast(K.argmax(y_pred, axis=-1),K.floatx())+1.0), K.floatx()) +     K.cast(K.equal(K.cast(K.argmax(y_true, axis=-1), K.floatx()), K.cast(K.argmax(y_pred, axis=-1),K.floatx())-2.0), K.floatx()) +     K.cast(K.equal(K.cast(K.argmax(y_true, axis=-1), K.floatx()), K.cast(K.argmax(y_pred, axis=-1),K.floatx())+2.0), K.floatx())    

# For reference, from keras documentation: https://github.com/keras-team/keras/blob/master/keras/losses.py
#def class_categorical_accuracy(y_true, y_pred):
    #return K.cast(K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred, axis=-1)),K.floatx())


# #### Configure model loss and optimization function

# In[31]:


# Define Optimizer
if optimizer_type == 'adam':
    model_optimizer = optimizers.Adam(lr = learning_rate) #, decay, beta_1, beta_2 are HPs
elif optimizer_type == 'rmsprop':
    model_optimizer = optimizers.RMSprop(lr = learning_rate) #, decay, rho
elif optimizer_type == 'gradient':
    model_optimizer = optimizers.SGD(lr = learning_rate) #, decay, momentum

# Compile model with appropriate loss function
if speed_bucket_size != 'none_use_regression': # if performing classification, ALWAYS use cross-entropy loss
    model.compile(loss ='categorical_crossentropy', optimizer=model_optimizer, metrics=['accuracy',class_percent_1buckRange,class_percent_2buckRange, class_mae, class_mse]) # class_percent_1buckLow,class_percent_1buckHigh,class_percent_2buckLow, class_percent_2buckHigh,'class_mape'
else:                                          # if performing regression, use mean squared error or mean absolute error
    if loss_function == 'categorical_crossentropy': raise NameError('Are you sure you want to use cross entropy loss with a regression tasks!?')
    model.compile(loss = loss_function, optimizer=model_optimizer, metrics=['mse','mae']) # options: 'mse','mae', 'mape'


# #### Train!

# In[32]:


# If desired, load weights from a previous model to start with model

if previous_model_weights_to_load != "":
    model.load_weights(folder_head_loc + "Model Final Parameters/" + previous_model_weights_to_load)


# In[33]:


# Show progress if running on local, only show final epoch values when running on Sherlock

if machine_to_run_script == 'local':
    verbose_option = 1 # show progress bar
elif machine_to_run_script == 'Sherlock':
    verbose_option = 2 # one line per epoch # 0 stay silent


# In[36]:


start_time = time.time()

if  model_architecture == 'FCN':
    history = model.fit(X_train, y_train, batch_size= batch_size, epochs=training_epochs, verbose=verbose_option, validation_data=(X_test, y_test))
elif model_architecture == 'CNN':
    history = model.fit([X_train_timeseries, X_train_anthro], y_train, batch_size= batch_size, epochs=training_epochs, verbose=verbose_option, validation_data=([X_test_timeseries, X_test_anthro], y_test))
    
end_time=time.time()


# In[ ]:


print('Finished training model!')


# ### Plot and save results

# #### Save a plot of results

# In[37]:


# Transform key results into a np arrary
trainAccuracy_1 = np.squeeze(history.history[accuracy_reporting_metric_1])
devAccuracy_1 = np.squeeze(history.history[dev_reporting_metric_1])
trainAccuracy_2 = np.squeeze(history.history[accuracy_reporting_metric_2])
devAccuracy_2 = np.squeeze(history.history[dev_reporting_metric_2])    
trainAccuracy_3 = np.squeeze(history.history[accuracy_reporting_metric_3])
devAccuracy_3 = np.squeeze(history.history[dev_reporting_metric_3])
trainAccuracy_4 = np.squeeze(history.history['acc'])
devAccuracy_4 = np.squeeze(history.history['val_acc'])
epochs = np.squeeze(range(1,training_epochs + 1))

# Save results to a .csv in the "Learning Curve Results"
df_devAccuracy = pd.DataFrame(np.transpose(np.vstack([epochs,devAccuracy_1, devAccuracy_2, devAccuracy_3, devAccuracy_4])))
filepath_acc = folder_head_loc + "Learning Curves/" + str(file_name) +"_AccuracyPerEpoch_Data" + ".csv"
df_devAccuracy.to_csv(filepath_acc, header = ["Epochs", dev_reporting_metric_1, dev_reporting_metric_2, dev_reporting_metric_3, 'acc'], index=False)

# Declare final values for results
final_accuracy_1 = history.history[accuracy_reporting_metric_1][training_epochs - 1]
final_accuracy_dev_1 = history.history[dev_reporting_metric_1][training_epochs - 1]
final_accuracy_2 = history.history[accuracy_reporting_metric_2][training_epochs - 1]
final_accuracy_dev_2 = history.history[dev_reporting_metric_2][training_epochs - 1]
final_accuracy_3 = history.history[accuracy_reporting_metric_3][training_epochs - 1]
final_accuracy_dev_3 = history.history[dev_reporting_metric_3][training_epochs - 1]
final_accuracy_4 = history.history['acc'][training_epochs - 1]
final_accuracy_dev_4 = history.history['val_acc'][training_epochs - 1]


# In[43]:


from matplotlib.patches import Rectangle
from matplotlib.legend import Legend

if machine_to_run_script == 'local':
    fig, ax1 = plt.subplots()
    lines=[]

    lines += ax1.plot(trainAccuracy_4,'#0e128c', label='Train Accuracy 2', linewidth=1) #'#DAF7A6'
    lines += ax1.plot(devAccuracy_4,'#a3a4cc', label='Dev Accuracy 2', linewidth=1)# '#33FF00',
    lines += ax1.plot(trainAccuracy_1,'#FF5733', label='Train Accuracy 1', linewidth=1)
    lines += ax1.plot(devAccuracy_1,'#C70039', label='Dev Accuracy 1', linewidth=1)
    lines += ax1.plot(trainAccuracy_2,'#9C27B0', label='Train Accuracy 2', linewidth=1)
    lines += ax1.plot(devAccuracy_2,'#7986CB', label='Dev Accuracy 2', linewidth=1)
    plt.ylim([0.0, 1.0])  # Surpress this for non-classification tasks

    plt.ylabel('Train vs. Dev Accuracy')

    plt.xlabel('Epochs')
    plt.title(dev_reporting_metric_1 + ": " + str(np.around(final_accuracy_dev_1,4)) + "\n" +              dev_reporting_metric_2 + ": " + str(np.around(final_accuracy_dev_2,4))) 
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    plt.legend([extra,extra,extra,extra,extra,extra,extra,extra,extra,extra],(
                                                    "loss: " + loss_function,
                                                    "learning rate: " + str(learning_rate),
                                                    "batch_size: " + str(batch_size),
                                                    "speed_bucket_size: " + speed_bucket_size,
                                                    "epochs: "+str(training_epochs),
                                                    "input_window_size: " + str(input_window_size),
                                                    "num_labels: " + str(len(labels_to_number)),
                                                    "evaluation metric 1:"+accuracy_reporting_metric_1,
                                                    "evaluation metric 2:"+accuracy_reporting_metric_2,
                                                    "evaluation metric 3:"+accuracy_reporting_metric_3,
                                                    "note:" + plot_note),
                                                    bbox_to_anchor=(1.05, 1),
                                                    loc=2,
                                                    borderaxespad=0.)

    leg = Legend(ax1, lines[0:], ['Train ACC', 'Dev ACC','Train Eval 1','Dev Eval 1'],
                 loc='best', frameon=False)
    ax1.add_artist(leg);
    plt.savefig(folder_head_loc + "Learning Curves/" + str(file_name) + "_AccuracyPerEpoch_Image.png", bbox_inches = "tight")
    plt.show()


# #### Record results of the model in a table

# In[39]:


# Add the results of the most recent run to the results file for documentation

if  model_architecture == 'FCN':
    a=[[model_architecture,
        file_name,
        "na",
        myFileLocation,
        training_epochs,  
        end_time - start_time,
        final_accuracy_1,
        final_accuracy_dev_1,
        final_accuracy_2,
        final_accuracy_dev_2,
        final_accuracy_3,
        final_accuracy_dev_3,
        batch_size,    
        learning_rate,
        speed_bucket_size,
        loss_function,
        input_window_size,
        label_window_size,
        optimizer_type,
        accuracy_reporting_metric_1,
        accuracy_reporting_metric_2,
        accuracy_reporting_metric_3,
        num_hidden_fc_layers,
        hidden_units_strategy,
        activations_strategy,
        dropout_rates
       ]]
    df=pd.DataFrame(a, columns=["model type",
                                "model filename",
                                "plot filename",
                                "data filename",
                                "epochs",
                                "runtime",
                                "train accuracy 1",
                                "dev accuracy 1",
                                "train accuracy 2",
                                "dev accuracy 2",
                                "train accuracy 3",
                                "dev accuracy 3",
                                "batch_size",  
                                "learning_rate",
                                "speed_bucket_size",
                                "loss_function",
                                "input_window_size",
                                "label_window_size",
                                "optimizer_type",
                                "evaluation_metric_1",
                                "evaluation_metric_2",
                                "evaluation_metric_3",
                                "num_hidden_fc_layers",
                                "hidden_units_strategy",
                                "activations_strategy",
                                "dropout_rates"])
    past_results = load_results_file_FCN(results_file_name)
elif model_architecture == 'CNN':     
    a=[[model_architecture,
        file_name,
        "na",
        myFileLocation,
        training_epochs,
        end_time - start_time,
        final_accuracy_1,
        final_accuracy_dev_1,
        final_accuracy_2,
        final_accuracy_dev_2,
        final_accuracy_3,
        final_accuracy_dev_3,
        batch_size,
        learning_rate,
        speed_bucket_size,
        loss_function,
        input_window_size,
        label_window_size,
        optimizer_type,
        accuracy_reporting_metric_1,
        accuracy_reporting_metric_2,
        accuracy_reporting_metric_3,
        hidden_units_strategy_CNN,
        num_filters,
        kernel_size,
        sample_stride,
        activation_conv_layer,
        activations_strategy_CNN,
        max_pool_kernel_size]]
    df=pd.DataFrame(a, columns=["model type",
                                "model filename",
                                "plot filename",
                                "data filename",
                                "epochs",
                                "runtime",
                                "dev accuracy 1",
                                "train accuracy 1",
                                "dev accuracy 2",
                                "train accuracy 2",
                                "dev accuracy 3",
                                "train accuracy 3",
                                "batch_size",
                                "learning_rate",
                                "speed_bucket_size",
                                "loss_function",
                                "input_window_size",
                                "label_window_size",
                                "optimizer_type",
                                "evaluation_metric_1",
                                "evaluation_metric_2",
                                "evaluation_metric_3",
                                "hidden_units_strategy_CNN",
                                "num_filters",
                                "kernel_size",
                                "sample_stride",
                                "activation_conv_layer",
                                "activations_strategy_CNN",
                                "max_pool_kernel_size"])    
    past_results = load_results_file_CNN(results_file_name)
past_results=pd.concat([past_results,df], sort=True)          # changed to sort=True on 20180907 based on docs to mute warning
#if machine_to_run_script == 'local':
    #print(past_results)
past_results.to_csv(folder_head_loc + "Model Performance Tables/" + results_file_name + ".csv",index=False ) # Consider fixing to put the columns in not-alphabetical order


# #### Build confusion matrix and regression plot

# In[40]:


if  model_architecture == 'FCN':
    y_pred = model.predict(X_test)
elif model_architecture == 'CNN':
    y_pred = model.predict([X_test_timeseries, X_test_anthro])
y_pred_argmax = np.argmax(y_pred, axis=1)

y_true = y_test
y_true_argmax = np.argmax(y_true, axis=1)


# In[41]:


# Plot results
if machine_to_run_script == 'local':
    plt.scatter(y_true_argmax, y_pred_argmax, s=3, alpha=0.3)
    plt.scatter(y_true_argmax, y_true_argmax, s=3, alpha=1)
    #plt.scatter(y_true, y_pred, s=3, alpha=0.3) # For regression
    plt.xlim([0,50])
    plt.ylim([0,50])
    plt.xlabel('Y_True')
    plt.ylabel('Y_Prediction')
    plt.savefig(folder_head_loc + "Confusion Matrices/" + str(file_name) + "_ConfusionMatrix_Image.png")
    plt.show()

# Record data in a .csv
y_trueVy_pred = np.vstack([y_true_argmax,y_pred_argmax])
df_y_trueVy_pred = pd.DataFrame(np.transpose(y_trueVy_pred))
filepath_predictions = folder_head_loc + "Model Final Predictions/" + str(file_name) + "_Predictions" + ".csv"
df_y_trueVy_pred.to_csv(filepath_predictions, header = ["y_true_argmax", "y_pred_argmax"], index=False)

# Create and save a confusion matrix
cm = confusion_matrix(y_true_argmax, y_pred_argmax)
df_cm = pd.DataFrame (cm)
filepath_cm = folder_head_loc + "Confusion Matrices/" + str(file_name) + "_ConfusionMatrix_Data.xlsx"
df_cm.to_excel(filepath_cm, index=False)


# In[ ]:


print('Successfully created plots and figures!')


# #### Save model parameters (weights)

# In[42]:


if machine_to_run_script == 'local':
    completed_model_name = file_name + "_" + model_architecture
    model.save_weights(folder_head_loc + "Model Final Parameters/" + completed_model_name + '_weights.h5')
    # THIS DOES NOT WORK IN SHERLOCK RIGHT NOW

# File "SpeedPrediction_SHERLOCK_20180807_v3.py", line 868, in <module>
#     model.save_weights(folder_head_loc + "Model Final Parameters/" + completed_model_name + '_weights.h5')
#   File "/share/software/user/open/py-keras/2.1.5_py36/lib/python3.6/site-packages/keras/engine/topology.py", line 2604, in save_weights
#     raise ImportError('`save_weights` requires h5py.')

# The following have been reloaded with a version change:
#   1) py-numpy/1.14.3_py36 => py-numpy/1.14.3_py27
#   2) python/3.6.1 => python/2.7.13


# ### End of Script
