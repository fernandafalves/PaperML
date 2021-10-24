# Time series forecasting based on historical data

from pandas import read_csv
from math import sqrt
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import pandas as pd
import random
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from numpy import array
from numpy import hstack
import sys
np.set_printoptions(threshold=sys.maxsize)
from sklearn.preprocessing import StandardScaler
from pandas import Series
from tensorflow.keras import regularizers
import joblib

for m in range(1,3):
    SEED = 123
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    if m == 1:
        dataset = read_csv('DATASET_MACHINE_1_FEATURES.csv', header = 0)
    else:
        dataset = read_csv('DATASET_MACHINE_2_FEATURES.csv', header = 0)

    del dataset['Period']

    train, test = dataset[0:8602], dataset[8602:10752]

    train = np.asarray(train)
    test = np.asarray(test)

    def scale(train, test):
        train = train.reshape(-1, 1)
        test = test.reshape(-1, 1)

        # fit scaler
        scaler = MinMaxScaler()
        train = train.astype(float)
        test = test.astype(float)
        scaler = scaler.fit(train)  
        # transform train
        train_scaled = scaler.transform(train)
        
        # transform test
        test_scaled = scaler.transform(test)
        return scaler, train_scaled, test_scaled

    train_Pressure = train[:,0]
    train_Pressure = train_Pressure.reshape(-1, 1)

    test_Pressure = test[:,0]
    test_Pressure = test_Pressure.reshape(-1, 1)

    scaler_Pressure, train_Pressure, test_Pressure = scale(train_Pressure,test_Pressure)

    #---

    train_Speed = train[:,1]
    train_Speed = train_Speed.reshape(-1, 1)

    test_Speed = test[:,1]
    test_Speed = test_Speed.reshape(-1, 1)

    scaler_Speed, train_Speed, test_Speed = scale(train_Speed,test_Speed)

    #---

    train_Temperature = train[:,2]
    train_Temperature = train_Temperature.reshape(-1, 1)

    test_Temperature = test[:,2]
    test_Temperature = test_Temperature.reshape(-1, 1)

    scaler_Temperature, train_Temperature, test_Temperature = scale(train_Temperature,test_Temperature)

    #---

    train_Sound = train[:,3]
    train_Sound = train_Sound.reshape(-1, 1)

    test_Sound = test[:,3]
    test_Sound = test_Sound.reshape(-1, 1)

    scaler_Sound, train_Sound, test_Sound = scale(train_Sound,test_Sound)

    #---

    train_Vibration = train[:,4]
    train_Vibration = train_Vibration.reshape(-1, 1)

    test_Vibration = test[:,4]
    test_Vibration = test_Vibration.reshape(-1, 1)

    scaler_Vibration, train_Vibration, test_Vibration = scale(train_Vibration,test_Vibration)

   
    # split a multivariate sequence into samples
    def split_sequences(sequences, n_steps_in, n_steps_out):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the dataset
            if out_end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    # Training set
    # define input sequence
    in_seq1 = array(train_Pressure)
    in_seq2 = array(train_Speed)
    in_seq3 = array(train_Temperature)
    in_seq4 = array(train_Sound)
    in_seq5 = array(train_Vibration)

    # horizontally stack columns
    dataset_mod = hstack((in_seq1, in_seq2, in_seq3, in_seq4, in_seq5))

    # choose a number of time steps
    n_steps_in, n_steps_out = 112, 1

    # covert into input/output
    X, y = split_sequences(dataset_mod, n_steps_in, n_steps_out)

    n_features = X.shape[2]

    # Test set
    # define input sequence
    in_seq1_test = array(test_Pressure)
    in_seq2_test = array(test_Speed)
    in_seq3_test = array(test_Temperature)
    in_seq4_test = array(test_Sound)
    in_seq5_test = array(test_Vibration)

    # horizontally stack columns
    dataset_test = hstack((in_seq1_test, in_seq2_test, in_seq3_test, in_seq4_test, in_seq5_test))

    # covert into input/output
    test_X, test_y = split_sequences(dataset_test, n_steps_in, n_steps_out)

    # define model
    if m == 1:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(32, kernel_regularizer = regularizers.l1(l1 = 0.00001), input_shape= (None, n_features), return_sequences = True))
        model.add(tf.keras.layers.LeakyReLU(alpha = 0.5)) 
        model.add(tf.keras.layers.LSTM(32, kernel_regularizer = regularizers.l1(l1 = 0.00001), return_sequences = True))
        model.add(tf.keras.layers.LeakyReLU(alpha = 0.5)) 
        model.add(tf.keras.layers.LSTM(32, kernel_regularizer = regularizers.l1(l1 = 0.00001), return_sequences = True))
        model.add(tf.keras.layers.LSTM(32, kernel_regularizer = regularizers.l1(l1 = 0.00001), return_sequences = False))
        model.add(tf.keras.layers.Dense(5))
    else:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(64, kernel_regularizer = regularizers.l1(l1 = 0.00001), input_shape= (None, n_features), return_sequences = True))
        model.add(tf.keras.layers.LSTM(64, kernel_regularizer = regularizers.l1(l1 = 0.00001), return_sequences = True))
        model.add(tf.keras.layers.LSTM(64, kernel_regularizer = regularizers.l1(l1 = 0.00001), return_sequences = True))
        model.add(tf.keras.layers.LSTM(64, kernel_regularizer = regularizers.l1(l1 = 0.00001), return_sequences = False))
        model.add(tf.keras.layers.Dense(5))    

    model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001), loss = 'mae')

    # fit model
    history = model.fit(X, y, epochs = 100, steps_per_epoch = 1500, verbose = 1, batch_size = 1, validation_data = (test_X, test_y), shuffle = False)

    if m == 1:
        model.save("final_forecasting_M1")
        model.save_weights("weights_forecasting_M1.h5")
    else:
        model.save("final_forecasting_M2")
        model.save_weights("weights_forecasting_M2.h5")

    # summarize history for loss
    #plt.rcParams.update({'font.size': 14})
    #plt.rcParams["font.family"] = "Times New Roman"
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.ylabel('Loss')
    #plt.xlabel('Epoch')
    #plt.legend(['Train', 'Test'], loc='upper right')
    #plt.show()

    if m == 1:
        newmodel = tf.keras.Sequential()
        newmodel.add(tf.keras.layers.LSTM(32, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, batch_input_shape= (1, None, n_features), return_sequences = True))
        newmodel.add(tf.keras.layers.LeakyReLU(alpha = 0.5)) 
        newmodel.add(tf.keras.layers.LSTM(32, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, return_sequences = True))
        newmodel.add(tf.keras.layers.LeakyReLU(alpha = 0.5)) 
        newmodel.add(tf.keras.layers.LSTM(32, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, return_sequences = True))
        newmodel.add(tf.keras.layers.LSTM(32, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, return_sequences = False))
        newmodel.add(tf.keras.layers.Dense(5))
    else:
        newmodel = tf.keras.Sequential()
        newmodel.add(tf.keras.layers.LSTM(64, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, batch_input_shape= (1, None, n_features), return_sequences = True))
        newmodel.add(tf.keras.layers.LSTM(64, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, return_sequences = True))
        newmodel.add(tf.keras.layers.LSTM(64, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, return_sequences = True))
        newmodel.add(tf.keras.layers.LSTM(64, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, return_sequences = False))
        newmodel.add(tf.keras.layers.Dense(5))

    x_input_pred = list()
    for i in range(len(dataset_test) - n_steps_in, len(dataset_test)):
        x_input_pred.append(dataset_test[i])

    x_input_pred = array(x_input_pred)
    x_input_pred = x_input_pred.reshape((1,n_steps_in, n_features))

    n_future_pred = 112

    x_input_pred = np.asarray(x_input_pred).astype(np.float32)

    newmodel.set_weights(model.get_weights())

    x_input_pred_II = x_input_pred

    predictions = []
    for i in range(n_future_pred):
        predictions = np.append(predictions, newmodel.predict(x_input_pred))
        predictions = np.asarray(predictions).reshape((1, 1, 1 + i, n_features))
        x_input_pred = np.append(x_input_pred_II, predictions[-1], axis = 1)

    predictions = array(predictions)
    predictions = np.reshape(predictions, (n_future_pred, n_features))

    if m == 1:
        real_data = read_csv('DATASET_MACHINE_1_FEATURES.csv', header = 0)
    else:
        real_data = read_csv('DATASET_MACHINE_2_FEATURES.csv', header = 0)

    pred_Pressure = predictions[:,0]
    pred_Pressure = pred_Pressure.reshape(-1, 1)

    pred_Speed = predictions[:,1]
    pred_Speed = pred_Speed.reshape(-1, 1)

    pred_Temperature = predictions[:,2]
    pred_Temperature = pred_Temperature.reshape(-1, 1)

    pred_Sound = predictions[:,3]
    pred_Sound = pred_Sound.reshape(-1, 1)

    pred_Vibration = predictions[:,4]
    pred_Vibration = pred_Vibration.reshape(-1, 1)

    predictions_Pressure = scaler_Pressure.inverse_transform(pred_Pressure)
    predictions_Speed = scaler_Speed.inverse_transform(pred_Speed)
    predictions_Temp = scaler_Temperature.inverse_transform(pred_Temperature)
    predictions_Sound = scaler_Sound.inverse_transform(pred_Sound)
    predictions_Vibration = scaler_Vibration.inverse_transform(pred_Vibration)

    if m == 1:
        new_data = read_csv('DATASET_MACHINE_1_FEATURES_1.csv', header = 0)
    else:
        new_data = read_csv('DATASET_MACHINE_2_FEATURES_1.csv', header = 0)    

    def RMSE_calculation(pred,actual):
        data = list()
        nd = array(actual)
        for i in range(0,112):
            data.append(nd[i])

        mse = mean_squared_error(data, pred)
        rmse = sqrt(mse)
        return rmse

    score_Pressure = RMSE_calculation(predictions_Pressure, new_data['Pressure'].values)        
    score_Speed = RMSE_calculation(predictions_Speed, new_data['Speed'].values)
    score_Temp = RMSE_calculation(predictions_Temp, new_data['Temperature'].values)
    score_Sound = RMSE_calculation(predictions_Sound, new_data['Sound'].values)
    score_Vibration = RMSE_calculation(predictions_Vibration, new_data['Vibration'].values)

    if m == 1:
        results = open("Forecasting_M1.txt","w") 
    else:
        results = open("Forecasting_M2.txt","w") 
    np.savetxt(results, predictions_Pressure)
    np.savetxt(results, predictions_Speed)
    np.savetxt(results, predictions_Temp)
    np.savetxt(results, predictions_Sound)
    np.savetxt(results, predictions_Vibration)

    results.close()

    results_scores = open("Results_Forecasting.txt","w") 
    results_scores.write("Period 1: \n")
    results_scores.write("RMSE Pressure: %.4f \n" % score_Pressure) 
    results_scores.write("RMSE Speed: %.4f \n" % score_Speed) 
    results_scores.write("RMSE Temperature: %.4f \n" % score_Temp) 
    results_scores.write("RMSE Sound: %.4f \n" % score_Sound) 
    results_scores.write("RMSE Vibration: %.4f \n" % score_Vibration) 
    results_scores.close() 