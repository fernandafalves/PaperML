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
from keras.layers import Dense
import pandas as pd
import random
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from numpy import array
from numpy import hstack
import joblib
import sys
np.set_printoptions(threshold=sys.maxsize)

for m in range(1,3):
    SEED = 123
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    if m == 1:
        arq = open('Forecasting_M1.txt', 'r')
    else:
        arq = open('Forecasting_M2.txt', 'r') 
    text = arq.read().splitlines()
    i = 0
    predictions_Pressure = []
    predictions_Speed = []
    predictions_Temp = []
    predictions_Sound = []
    predictions_Vibration = []

    for line in text:
        if i < 112:
            predictions_Pressure = np.append(predictions_Pressure, line)
        else:
            if i >= 112 and i < 224:
                predictions_Speed = np.append(predictions_Speed, line)    
            else:
                if i >= 224 and i < 336:
                    predictions_Temp = np.append(predictions_Temp, line)    
                else:
                    if i >= 336 and i < 448:
                        predictions_Sound = np.append(predictions_Sound, line)        
                    else:
                        predictions_Vibration = np.append(predictions_Vibration, line)                   
        i = i + 1
    arq.close()

    predictions_Pressure = np.asarray(predictions_Pressure, dtype = 'float64')
    predictions_Speed = np.asarray(predictions_Speed, dtype = 'float64')
    predictions_Temp = np.asarray(predictions_Temp, dtype = 'float64')
    predictions_Sound = np.asarray(predictions_Sound, dtype = 'float64')
    predictions_Vibration = np.asarray(predictions_Vibration, dtype = 'float64')

    predictions_Pressure = np.around(predictions_Pressure, decimals = 2, out = None)
    predictions_Speed = np.around(predictions_Speed, decimals = 2, out = None)
    predictions_Temp = np.around(predictions_Temp, decimals = 2, out = None)
    predictions_Sound = np.around(predictions_Sound, decimals = 2, out = None)
    predictions_Vibration = np.around(predictions_Vibration, decimals = 2, out = None)

    dataset_class = pd.DataFrame(list(zip(predictions_Pressure, predictions_Speed, predictions_Temp, predictions_Sound, predictions_Vibration)), columns =['Pressure', 'Speed', 'Temperature', 'Sound', 'Vibration'])
    if m == 1:
        dataset_failures = read_csv('DATASET_MACHINE_1.csv', header = 0)
    else:
        dataset_failures = read_csv('DATASET_MACHINE_2.csv', header = 0)

    del dataset_failures['Period']

    # Class Distribution
    class_counts = dataset_failures.groupby('Failures').size()
    print(class_counts)

    train, test = dataset_failures[0:8602], dataset_failures[8602:10752]

    train = np.asarray(train)
    test = np.asarray(test)

    def scale(train, test):
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

    input_Failures = train[:,0]
    input_Failures = input_Failures.reshape(-1, 1)

    input_Failures_test = test[:,0]
    input_Failures_test = input_Failures_test.reshape(-1, 1)

    #---

    input_Pressure = train[:,1]
    input_Pressure = input_Pressure.reshape(-1, 1)

    input_Pressure_test = test[:,1]
    input_Pressure_test = input_Pressure_test.reshape(-1, 1)

    scaler_Pressure, input_Pressure, input_Pressure_test = scale(input_Pressure, input_Pressure_test)

    #---

    input_Speed = train[:,2]
    input_Speed = input_Speed.reshape(-1, 1)

    input_Speed_test = test[:,2]
    input_Speed_test = input_Speed_test.reshape(-1, 1)

    scaler_Speed, input_Speed, input_Speed_test = scale(input_Speed, input_Speed_test)

    #---

    input_Temperature = train[:,3]
    input_Temperature = input_Temperature.reshape(-1, 1)

    input_Temperature_test = test[:,3]
    input_Temperature_test = input_Temperature_test.reshape(-1, 1)

    scaler_Temperature, input_Temperature, input_Temperature_test = scale(input_Temperature, input_Temperature_test)

    #---

    input_Sound = train[:,4]
    input_Sound = input_Sound.reshape(-1, 1)

    input_Sound_test = test[:,4]
    input_Sound_test = input_Sound_test.reshape(-1, 1)

    scaler_Sound, input_Sound, input_Sound_test = scale(input_Sound, input_Sound_test)

    #---

    input_Vibration = train[:,5]
    input_Vibration = input_Vibration.reshape(-1, 1)

    input_Vibration_test = test[:,5]
    input_Vibration_test = input_Vibration_test.reshape(-1, 1)

    scaler_Vibration, input_Vibration, input_Vibration_test = scale(input_Vibration, input_Vibration_test)

    #---

    X = pd.DataFrame(list(zip(array(input_Pressure), array(input_Sound), array(input_Speed), array(input_Temperature), array(input_Vibration))), columns =['Pressure', 'Speed', 'Temperature', 'Sound', 'Vibration'])
    Y = array(input_Failures)
    
    X_test = pd.DataFrame(list(zip(array(input_Pressure_test), array(input_Sound_test), array(input_Speed_test), array(input_Temperature_test), array(input_Vibration_test))), columns =['Pressure', 'Speed', 'Temperature', 'Sound', 'Vibration'])

    Y_test = array(input_Failures_test)

    X = np.asarray(X).astype('float32')
    X_test = np.asarray(X_test).astype('float32')

    # CLASSIFICATION OF THE FAILURES - TRAINING

    model = Sequential()
    model.add(Dense(256, input_dim = 5, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))    

    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = keras.metrics.AUC(name = 'auc'))
    
    model.fit(X, Y, epochs = 40, batch_size = 10, verbose = 1, validation_data = (X_test, Y_test))

    if m == 1:
        model.save("final_model_M1")
        model.save_weights("weights_M1.h5")
    else:
        model.save("final_model_M2")
        model.save_weights("weights_M2.h5")   