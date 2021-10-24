import amplpy
from pandas import read_csv
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import random
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from numpy import array
import joblib
import sys
np.set_printoptions(threshold=sys.maxsize)
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from tensorflow.keras import regularizers
from numpy import hstack
import time

from amplpy import AMPL, Environment
ampl = AMPL()

def scale_data(data):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = data.astype(float)
    scaler = scaler.fit(data)  
    data_scaled = scaler.transform(data)
    return scaler, data_scaled

def save_predictions(predictions_Pressure, predictions_Speed, predictions_Temp, predictions_Sound, predictions_Vibration):
    if m == 1:
        Pressure = ampl.getSet("Pressure")
        Pressure.setValues(predictions_Pressure)

        Speed = ampl.getSet("Speed")
        Speed.setValues(predictions_Speed)

        Temperature = ampl.getSet("Temperature")
        Temperature.setValues(predictions_Temp)

        Sound = ampl.getSet("Sound")
        Sound.setValues(predictions_Sound)

        Vibration = ampl.getSet("Vibration")
        Vibration.setValues(predictions_Vibration)
    else:
        Pressure_M2 = ampl.getSet("Pressure_M2")
        Pressure_M2.setValues(predictions_Pressure)

        Speed_M2 = ampl.getSet("Speed_M2")
        Speed_M2.setValues(predictions_Speed)

        Temperature_M2 = ampl.getSet("Temperature_M2")
        Temperature_M2.setValues(predictions_Temp)

        Sound_M2 = ampl.getSet("Sound_M2")
        Sound_M2.setValues(predictions_Sound)

        Vibration_M2 = ampl.getSet("Vibration_M2")
        Vibration_M2.setValues(predictions_Vibration)

def save_reference(reference_Pressure, reference_Speed, reference_Temp, reference_Sound, reference_Vibration):
    if m == 1:
        Reference_Pressure = ampl.getSet("Reference_Pressure")
        Reference_Pressure.setValues(reference_Pressure)

        Reference_Speed = ampl.getSet("Reference_Speed")
        Reference_Speed.setValues(reference_Speed)

        Reference_Temperature = ampl.getSet("Reference_Temperature")
        Reference_Temperature.setValues(reference_Temp)

        Reference_Sound = ampl.getSet("Reference_Sound")
        Reference_Sound.setValues(reference_Sound)

        Reference_Vibration = ampl.getSet("Reference_Vibration")
        Reference_Vibration.setValues(reference_Vibration)
    else:
        Reference_Pressure_M2 = ampl.getSet("Reference_Pressure_M2")
        Reference_Pressure_M2.setValues(reference_Pressure)

        Reference_Speed_M2 = ampl.getSet("Reference_Speed_M2")
        Reference_Speed_M2.setValues(reference_Speed)

        Reference_Temperature_M2 = ampl.getSet("Reference_Temperature_M2")
        Reference_Temperature_M2.setValues(reference_Temp)

        Reference_Sound_M2 = ampl.getSet("Reference_Sound_M2")
        Reference_Sound_M2.setValues(reference_Sound)

        Reference_Vibration_M2 = ampl.getSet("Reference_Vibration_M2")
        Reference_Vibration_M2.setValues(reference_Vibration)

def transform_sets(reference_Pressure, reference_Speed, reference_Temp, reference_Sound, reference_Vibration):
    # To avoid repeated numbers in the sets of 'save_reference'
    set_Pressure = set()
    counter = 0
    for i in range(0,112):
        if reference_Pressure.item(i) not in set_Pressure:
            set_Pressure.add(reference_Pressure.item(i))
        else:
            counter += 0.0000001
            reference_Pressure[[i]] = reference_Pressure[[i]] + counter   

    set_Speed = set()
    counter = 0
    for i in range(0,112):
        if reference_Speed.item(i) not in set_Speed:
            set_Speed.add(reference_Speed.item(i))
        else:
            counter += 0.0000001
            reference_Speed[[i]] = reference_Speed[[i]] + counter

    set_Temp = set()
    counter = 0
    for i in range(0,112):
        if reference_Temp.item(i) not in set_Temp:
            set_Temp.add(reference_Temp.item(i))
        else:
            counter += 0.0000001
            reference_Temp[[i]] = reference_Temp[[i]] + counter    

    set_Sound = set()
    counter = 0
    for i in range(0,112):
        if reference_Sound.item(i) not in set_Sound:
            set_Sound.add(reference_Sound.item(i))
        else:
            counter += 0.0000001
            reference_Sound[[i]] = reference_Sound[[i]] + counter   

    set_Vibration = set()
    counter = 0
    for i in range(0,112):
        if reference_Vibration.item(i) not in set_Vibration:
            set_Vibration.add(reference_Vibration.item(i))
        else:
            counter += 0.0000001
            reference_Vibration[[i]] = reference_Vibration[[i]] + counter

    return reference_Pressure, reference_Speed, reference_Temp, reference_Sound, reference_Vibration

for value in range(1,10):
    for variation in range(1,16):
        ini = time.time()
        SEED = 123
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        
        for time_period in range(1,3):
            

            if time_period == 1:
                ampl.eval('reset;')
                ampl.read('Data_ProactiveOnline_PMLALG_F2.txt')
            ampl.eval('let first_iteration:= 1;')
            ampl.eval('let subp:= 0;')

            if time_period == 1:
                for m in range(1,3):
                    SEED = 123
                    random.seed(SEED)
                    np.random.seed(SEED)
                    tf.random.set_seed(SEED)

                    if m == 1:
                        arq = open('Forecasting_M1.txt', 'r')
                        model = keras.models.load_model('final_model_M1')
                        model.load_weights("weights_M1.h5")
                    else:
                        arq = open('Forecasting_M2.txt', 'r') 
                        model = keras.models.load_model('final_model_M2')
                        model.load_weights("weights_M2.h5")

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

                    # More decimals to avoid repeated numbers in the sets of 'save_predictions'
                    predictions_Pressure = np.around(predictions_Pressure, decimals = 6, out = None)
                    predictions_Speed = np.around(predictions_Speed, decimals = 6, out = None)
                    predictions_Temp = np.around(predictions_Temp, decimals = 6, out = None)
                    predictions_Sound = np.around(predictions_Sound, decimals = 6, out = None)
                    predictions_Vibration = np.around(predictions_Vibration, decimals = 6, out = None)

                    if m == 1:
                        save_predictions(predictions_Pressure, predictions_Speed, predictions_Temp, predictions_Sound, predictions_Vibration)
                    else:
                        save_predictions(predictions_Pressure, predictions_Speed, predictions_Temp, predictions_Sound, predictions_Vibration)

                    predictions_Pressure = np.around(predictions_Pressure, decimals = 2, out = None)
                    predictions_Speed = np.around(predictions_Speed, decimals = 2, out = None)
                    predictions_Temp = np.around(predictions_Temp, decimals = 2, out = None)
                    predictions_Sound = np.around(predictions_Sound, decimals = 2, out = None)
                    predictions_Vibration = np.around(predictions_Vibration, decimals = 2, out = None)

                    dataset_class = pd.DataFrame(list(zip(predictions_Pressure, predictions_Speed, predictions_Temp, predictions_Sound, predictions_Vibration)), columns =['Pressure', 'Speed', 'Temperature', 'Sound', 'Vibration'])

                    input_Pressure_class = dataset_class['Pressure'].values
                    input_Pressure_class = input_Pressure_class.reshape(-1, 1)

                    scaler_Pressure_class, input_Pressure_class = scale_data(input_Pressure_class)

                    #---

                    input_Speed_class = dataset_class['Speed'].values
                    input_Speed_class = input_Speed_class.reshape(-1, 1)

                    scaler_Speed_class, input_Speed_class = scale_data(input_Speed_class)

                    #---

                    input_Temperature_class = dataset_class['Temperature'].values
                    input_Temperature_class = input_Temperature_class.reshape(-1, 1)

                    scaler_Temperature_class, input_Temperature_class = scale_data(input_Temperature_class)

                    #---

                    input_Sound_class = dataset_class['Sound'].values
                    input_Sound_class = input_Sound_class.reshape(-1, 1)

                    scaler_Sound_class, input_Sound_class = scale_data(input_Sound_class)

                    #---

                    input_Vibration_class = dataset_class['Vibration'].values
                    input_Vibration_class = input_Vibration_class.reshape(-1, 1)

                    scaler_Vibration_class, input_Vibration_class = scale_data(input_Vibration_class)

                    #---

                    X_class = pd.DataFrame(list(zip(array(input_Pressure_class), array(input_Sound_class), array(input_Speed_class), array(input_Temperature_class), array(input_Vibration_class))), columns =['Pressure', 'Speed', 'Temperature', 'Sound', 'Vibration'])
                    X_class = np.asarray(X_class).astype('float32')

                    # make probability predictions with the model
                    predictions = model(X_class)
                    
                    if m == 1:
                        pred_M1 = [np.round(x[0]) for x in predictions]
                        pred_M1 = np.asarray(pred_M1).astype('float32')

                        pred_M1 = pred_M1.reshape(-1, 1)

                        j = 0
                        for i in pred_M1:
                            if pred_M1[j] == 1 and pred_M1[j + 1] == 1:
                                pred_M1[j] = 0
                            j = j + 1
                    else:
                        pred_M2 = [np.round(x[0]) for x in predictions]
                        pred_M2 = np.asarray(pred_M2).astype('float32')

                        pred_M2 = pred_M2.reshape(-1, 1)

                        j = 0
                        for i in pred_M2:
                            if pred_M2[j] == 1 and pred_M2[j + 1] == 1:
                                pred_M2[j] = 0
                            j = j + 1
            else:
                if time_period == 2 and value == 1 and variation == 1:
                    for m in range(1,3):
                        SEED = 123
                        random.seed(SEED)
                        np.random.seed(SEED)
                        tf.random.set_seed(SEED)

                        if m == 1:
                            model = keras.models.load_model('final_forecasting_M1')
                            model.load_weights("weights_forecasting_M1.h5")
                            newmodel = tf.keras.Sequential()
                            newmodel.add(tf.keras.layers.LSTM(32, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, batch_input_shape= (1, None, 5), return_sequences = True))
                            newmodel.add(tf.keras.layers.LeakyReLU(alpha = 0.5)) 
                            newmodel.add(tf.keras.layers.LSTM(32, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, return_sequences = True))
                            newmodel.add(tf.keras.layers.LeakyReLU(alpha = 0.5)) 
                            newmodel.add(tf.keras.layers.LSTM(32, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, return_sequences = True))
                            newmodel.add(tf.keras.layers.LSTM(32, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, return_sequences = False))
                            newmodel.add(tf.keras.layers.Dense(5))

                            previous_real_failures = read_csv('DATASET_MACHINE_1_FEATURES_1.csv', header = 0)
                            del previous_real_failures['Period']
                        else:
                            model = keras.models.load_model('final_forecasting_M2')
                            model.load_weights("weights_forecasting_M2.h5")
                            newmodel = tf.keras.Sequential()
                            newmodel.add(tf.keras.layers.LSTM(64, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, batch_input_shape= (1, None, 5), return_sequences = True))
                            newmodel.add(tf.keras.layers.LSTM(64, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, return_sequences = True))
                            newmodel.add(tf.keras.layers.LSTM(64, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, return_sequences = True))
                            newmodel.add(tf.keras.layers.LSTM(64, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, return_sequences = False))
                            newmodel.add(tf.keras.layers.Dense(5))

                            previous_real_failures = read_csv('DATASET_MACHINE_2_FEATURES_1.csv', header = 0)
                            del previous_real_failures['Period']

                        # Using the saved time series forecasting model to forecast the features of period t = 2
                        previous_real_failures = array(previous_real_failures)

                        SEED = 123
                        random.seed(SEED)
                        np.random.seed(SEED)
                        tf.random.set_seed(SEED)

                        if m == 1:
                            dataset = read_csv('DATASET_MACHINE_1_FEATURES.csv', header = 0)
                        else:
                            dataset = read_csv('DATASET_MACHINE_2_FEATURES.csv', header = 0)

                        del dataset['Period']

                        train = dataset[0:8602]

                        train = np.asarray(train)

                        def scale(train, data):
                            # fit scaler
                            scaler = MinMaxScaler()
                            train = train.astype(float)
                            data = data.astype(float)
                            scaler = scaler.fit(train)
                            
                            # transform data
                            data_scaled = scaler.transform(data)
                            return scaler, data_scaled

                        train_Pressure = train[:,0]
                        train_Pressure = train_Pressure.reshape(-1, 1)

                        data_Pressure = previous_real_failures[:,0]
                        data_Pressure = data_Pressure.reshape(-1, 1)

                        scaler_Pressure, data_Pressure = scale(train_Pressure, data_Pressure)

                        #---

                        train_Speed = train[:,1]
                        train_Speed = train_Speed.reshape(-1, 1)

                        data_Speed = previous_real_failures[:,1]
                        data_Speed = data_Speed.reshape(-1, 1)

                        scaler_Speed, data_Speed = scale(train_Speed, data_Speed)

                        #---

                        train_Temperature = train[:,2]
                        train_Temperature = train_Temperature.reshape(-1, 1)

                        data_Temperature = previous_real_failures[:,2]
                        data_Temperature = data_Temperature.reshape(-1, 1)

                        scaler_Temperature, data_Temperature = scale(train_Temperature, data_Temperature)

                        #---

                        train_Sound = train[:,3]
                        train_Sound = train_Sound.reshape(-1, 1)

                        data_Sound = previous_real_failures[:,3]
                        data_Sound = data_Sound.reshape(-1, 1)

                        scaler_Sound, data_Sound = scale(train_Sound, data_Sound)

                        #---

                        train_Vibration = train[:,4]
                        train_Vibration = train_Vibration.reshape(-1, 1)

                        data_Vibration = previous_real_failures[:,4]
                        data_Vibration = data_Vibration.reshape(-1, 1)

                        scaler_Vibration, data_Vibration = scale(train_Vibration, data_Vibration)
                        
                        in_seq1 = array(data_Pressure)
                        in_seq2 = array(data_Speed)
                        in_seq3 = array(data_Temperature)
                        in_seq4 = array(data_Sound)
                        in_seq5 = array(data_Vibration)

                        previous_real_failures = hstack((in_seq1, in_seq2, in_seq3, in_seq4, in_seq5))
                        
                        x_input_pred = list()

                        for i in previous_real_failures:
                            x_input_pred.append(i)

                        x_input_pred = array(x_input_pred)
                        x_input_pred = x_input_pred.reshape((1,112, 5))

                        n_future_pred = 112

                        x_input_pred = np.asarray(x_input_pred).astype(np.float32)

                        newmodel.set_weights(model.get_weights())                        

                        x_input_pred_II = x_input_pred

                        predictions = []
                        for i in range(n_future_pred):
                            predictions = np.append(predictions, newmodel(x_input_pred))
                            predictions = np.asarray(predictions).reshape((1, 1, 1 + i, 5))
                            x_input_pred = np.append(x_input_pred_II, predictions[-1], axis = 1)

                        predictions = array(predictions)
                        predictions = np.reshape(predictions, (n_future_pred, 5))

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
                            save_predictions(predictions_Pressure, predictions_Speed, predictions_Temp, predictions_Sound, predictions_Vibration)
                        else:
                            save_predictions(predictions_Pressure, predictions_Speed, predictions_Temp, predictions_Sound, predictions_Vibration)

                        dataset_class = pd.DataFrame(list(zip(predictions_Pressure, predictions_Speed, predictions_Temp, predictions_Sound, predictions_Vibration)), columns =['Pressure', 'Speed', 'Temperature', 'Sound', 'Vibration'])

                        input_Pressure_class = dataset_class['Pressure'].values
                        input_Pressure_class = input_Pressure_class.reshape(-1, 1)

                        scaler_Pressure_class, input_Pressure_class = scale_data(input_Pressure_class)

                        #---

                        input_Speed_class = dataset_class['Speed'].values
                        input_Speed_class = input_Speed_class.reshape(-1, 1)

                        scaler_Speed_class, input_Speed_class = scale_data(input_Speed_class)

                        #---

                        input_Temperature_class = dataset_class['Temperature'].values
                        input_Temperature_class = input_Temperature_class.reshape(-1, 1)

                        scaler_Temperature_class, input_Temperature_class = scale_data(input_Temperature_class)

                        #---

                        input_Sound_class = dataset_class['Sound'].values
                        input_Sound_class = input_Sound_class.reshape(-1, 1)

                        scaler_Sound_class, input_Sound_class = scale_data(input_Sound_class)

                        #---

                        input_Vibration_class = dataset_class['Vibration'].values
                        input_Vibration_class = input_Vibration_class.reshape(-1, 1)

                        scaler_Vibration_class, input_Vibration_class = scale_data(input_Vibration_class)

                        #---

                        X_class = pd.DataFrame(list(zip(array(input_Pressure_class), array(input_Sound_class), array(input_Speed_class), array(input_Temperature_class), array(input_Vibration_class))), columns =['Pressure', 'Speed', 'Temperature', 'Sound', 'Vibration'])
                        X_class = np.asarray(X_class).astype('float32')

                        if m == 1:
                            classification_model = keras.models.load_model('final_model_M1')
                            classification_model.load_weights("weights_M1.h5")
                        else:
                            classification_model = keras.models.load_model('final_model_M2')
                            classification_model.load_weights("weights_M2.h5")

                        # make probability predictions with the model
                        predictions = classification_model(X_class)
                        
                        if m == 1:
                            pred_M1_2 = [np.round(x[0]) for x in predictions]
                            pred_M1_2 = np.asarray(pred_M1_2).astype('float32')

                            pred_M1_2 = pred_M1_2.reshape(-1, 1)

                            j = 0
                            for i in pred_M1_2:
                                if pred_M1_2[j] == 1 and pred_M1_2[j + 1] == 1:
                                    pred_M1_2[j] = 0
                                j = j + 1
                        else:
                            pred_M2_2 = [np.round(x[0]) for x in predictions]
                            pred_M2_2 = np.asarray(pred_M2_2).astype('float32')

                            pred_M2_2 = pred_M2_2.reshape(-1, 1)

                            j = 0
                            for i in pred_M2_2:
                                if pred_M2_2[j] == 1 and pred_M2_2[j + 1] == 1:
                                    pred_M2_2[j] = 0
                                j = j + 1               
                else:
                    if time_period == 2 and (value != 1 or variation != 1):
                        for m in range(1,3):
                            if m == 1:
                                results = open("/Forecastings/Forecasting_M1_%d.txt" % (time_period),"r") 
                            else:
                                results = open("/Forecastings/Forecasting_M2_%d.txt" % (time_period),"r") 

                            text = results.read().splitlines()
                            i = 0
                            predictions_Pressure = []
                            predictions_Speed = []
                            predictions_Temp = []
                            predictions_Sound = []
                            predictions_Vibration = []

                            number_data = 112

                            for line in text:
                                if i < number_data:
                                    predictions_Pressure = np.append(predictions_Pressure, line)
                                else:
                                    if i >= number_data and i < 2*number_data:
                                        predictions_Speed = np.append(predictions_Speed, line)    
                                    else:
                                        if i >= 2*number_data and i < 3*number_data:
                                            predictions_Temp = np.append(predictions_Temp, line)    
                                        else:
                                            if i >= 3*number_data and i < 4*number_data:
                                                predictions_Sound = np.append(predictions_Sound, line)        
                                            else:
                                                predictions_Vibration = np.append(predictions_Vibration, line)                   
                                i = i + 1
                            results.close()

                            predictions_Pressure = np.asarray(predictions_Pressure, dtype = 'float64')
                            predictions_Speed = np.asarray(predictions_Speed, dtype = 'float64')
                            predictions_Temp = np.asarray(predictions_Temp, dtype = 'float64')
                            predictions_Sound = np.asarray(predictions_Sound, dtype = 'float64')
                            predictions_Vibration = np.asarray(predictions_Vibration, dtype = 'float64')

                            # More decimals to avoid repeated numbers in the sets of 'save_predictions'
                            predictions_Pressure = np.around(predictions_Pressure, decimals = 6, out = None)
                            predictions_Speed = np.around(predictions_Speed, decimals = 6, out = None)
                            predictions_Temp = np.around(predictions_Temp, decimals = 6, out = None)
                            predictions_Sound = np.around(predictions_Sound, decimals = 6, out = None)
                            predictions_Vibration = np.around(predictions_Vibration, decimals = 6, out = None)

                            if m == 1:
                                ampl.eval('reset data Pressure, Speed, Temperature, Sound, Vibration;')
                                save_predictions(predictions_Pressure, predictions_Speed, predictions_Temp, predictions_Sound, predictions_Vibration)

                                predictions = open("/Forecastings/Classification_M1_%d.txt" % (time_period),"r") 
                            else:
                                ampl.eval('reset data Pressure_M2, Speed_M2, Temperature_M2, Sound_M2, Vibration_M2;')
                                save_predictions(predictions_Pressure, predictions_Speed, predictions_Temp, predictions_Sound, predictions_Vibration)

                                predictions = open("/Forecastings/Classification_M2_%d.txt" % (time_period),"r")

                            text = predictions.read().splitlines()
                            predictions = []

                            for line in text:
                                predictions = np.append(predictions, line)

                            predictions = np.asarray(predictions, dtype = 'float64')
                            predictions = predictions.reshape(-1, 1)     
                            
                            if m == 1:
                                pred_M1_2 = []
                                pred_M1_2 = [np.round(x[0]) for x in predictions]
                                pred_M1_2 = np.asarray(pred_M1_2).astype('float32')

                                pred_M1_2 = pred_M1_2.reshape(-1, 1)

                                j = 0
                                for i in pred_M1_2:
                                    if j + 1 < len(pred_M1_2):
                                        if pred_M1_2[j] == 1 and pred_M1_2[j + 1] == 1:
                                            pred_M1_2[j] = 0
                                    j = j + 1
                            else:
                                pred_M2_2 = []
                                pred_M2_2 = [np.round(x[0]) for x in predictions]
                                pred_M2_2 = np.asarray(pred_M2_2).astype('float32')

                                pred_M2_2 = pred_M2_2.reshape(-1, 1)

                                j = 0
                                for i in pred_M2_2:
                                    if j + 1 < len(pred_M2_2):
                                        if pred_M2_2[j] == 1 and pred_M2_2[j + 1] == 1:
                                            pred_M2_2[j] = 0
                                    j = j + 1

            k = 1
            j = 1

            MOMENTS_1 = list()
            MOMENTS_2 = list()
            
            if time_period == 1:
                for period in range(0,112):
                    if pred_M1[period] == 1:
                        MOMENTS_1.append(period + 1)
                        k = k + 1  
                    
                    if pred_M2[period] == 1:
                        MOMENTS_2.append(period + 1)
                        j = j + 1
            else:
                for period in range(0,112):
                    if pred_M1_2[period] == 1:
                        MOMENTS_1.append(period + 1)
                        k = k + 1  
                    
                    if pred_M2_2[period] == 1:
                        MOMENTS_2.append(period + 1)
                        j = j + 1

                # Prediction error
                error_M1 = (fp_M1/fn_M1)
                if error_M1 > 1:
                    error_M1 = error_M1 - 1
                else:
                    error_M1 = 1 - error_M1 

                error_M2 = (fp_M2/fn_M2)
                if error_M2 > 1:
                    error_M2 = error_M2 - 1
                else:
                    error_M2 = 1 - error_M2

                upper_bound_M1 = np.round(error_M1*fn_M1) 
                upper_bound_M2 = np.round(error_M2*fn_M2) 
                
                UB_M1 = ampl.getParameter("UB_M1")
                UB_M1.set(upper_bound_M1)

                UB_M2 = ampl.getParameter("UB_M2")
                UB_M2.set(upper_bound_M2) 


            if time_period == 1:
                reference_M1 = read_csv('DATASET_MACHINE_1_1.csv', header = 0)
                reference_M2 = read_csv('DATASET_MACHINE_2_1.csv', header = 0)
            else:
                reference_M1 = read_csv('DATASET_MACHINE_1_2.csv', header = 0)
                reference_M2 = read_csv('DATASET_MACHINE_2_2.csv', header = 0)

            reference_M1 = np.asarray(reference_M1)
            reference_M2 = np.asarray(reference_M2)

            for m in range(1,3):  
                if m == 1:  
                    reference_Pressure = reference_M1[:,2]
                    reference_Pressure = reference_Pressure.reshape(-1, 1)

                    reference_Speed = reference_M1[:,3]
                    reference_Speed = reference_Speed.reshape(-1, 1)

                    reference_Temp = reference_M1[:,4]
                    reference_Temp = reference_Temp.reshape(-1, 1)

                    reference_Sound = reference_M1[:,5]
                    reference_Sound = reference_Sound.reshape(-1, 1)

                    reference_Vibration = reference_M1[:,6]
                    reference_Vibration = reference_Vibration.reshape(-1, 1)
                                
                    reference_Pressure = array(reference_Pressure)
                    reference_Sound = array(reference_Sound)
                    reference_Temp = array(reference_Temp)
                    reference_Sound = array(reference_Sound)
                    reference_Vibration = array(reference_Vibration)

                    reference_Pressure, reference_Speed, reference_Temp, reference_Sound, reference_Vibration = transform_sets(reference_Pressure, reference_Speed, reference_Temp, reference_Sound, reference_Vibration)
                    
                    save_reference(reference_Pressure, reference_Speed, reference_Temp, reference_Sound, reference_Vibration)
                else:
                    reference_Pressure = reference_M2[:,2]
                    reference_Pressure = reference_Pressure.reshape(-1, 1)

                    reference_Speed = reference_M2[:,3]
                    reference_Speed = reference_Speed.reshape(-1, 1)

                    reference_Temp = reference_M2[:,4]
                    reference_Temp = reference_Temp.reshape(-1, 1)

                    reference_Sound = reference_M2[:,5]
                    reference_Sound = reference_Sound.reshape(-1, 1)

                    reference_Vibration = reference_M2[:,6]
                    reference_Vibration = reference_Vibration.reshape(-1, 1)
                                
                    reference_Pressure = array(reference_Pressure)
                    reference_Sound = array(reference_Sound)
                    reference_Temp = array(reference_Temp)
                    reference_Sound = array(reference_Sound)
                    reference_Vibration = array(reference_Vibration)

                    reference_Pressure, reference_Speed, reference_Temp, reference_Sound, reference_Vibration = transform_sets(reference_Pressure, reference_Speed, reference_Temp, reference_Sound, reference_Vibration)

                    save_reference(reference_Pressure, reference_Speed, reference_Temp, reference_Sound, reference_Vibration)

            if time_period == 1:
                reference_M1 = read_csv('DATASET_MACHINE_1_1.csv', header = 0)
                reference_M2 = read_csv('DATASET_MACHINE_2_1.csv', header = 0)
            else:
                reference_M1 = read_csv('DATASET_MACHINE_1_2.csv', header = 0)
                reference_M2 = read_csv('DATASET_MACHINE_2_2.csv', header = 0)

            del reference_M1['Period']
            del reference_M1['Pressure']
            del reference_M1['Speed']
            del reference_M1['Temperature']
            del reference_M1['Sound']
            del reference_M1['Vibration']

            del reference_M2['Period']
            del reference_M2['Pressure']
            del reference_M2['Speed']
            del reference_M2['Temperature']
            del reference_M2['Sound']
            del reference_M2['Vibration']

            reference_M1 = np.asarray(reference_M1)
            reference_M2 = np.asarray(reference_M2)

            reference_M1 = reference_M1[:,0]
            reference_M1 = reference_M1.reshape(-1, 1)

            reference_M2 = reference_M2[:,0]
            reference_M2 = reference_M2.reshape(-1, 1)

            k = 1
            j = 1

            REFERENCE_MOMENTS_1 = list()
            REFERENCE_MOMENTS_2 = list()
  
            for period in range(0,112):
                if reference_M1[period] == 1:
                    REFERENCE_MOMENTS_1.append(period + 1)
                    k = k + 1  
                
                if reference_M2[period] == 1:
                    REFERENCE_MOMENTS_2.append(period + 1)
                    j = j + 1

            ampl.eval('reset data Set_Moment_Failures_1, Set_Moment_Failures_2, Set_Reference_Failures_1, Set_Reference_Failures_2;')
            Set_Moment_Failures_1 = ampl.getSet("Set_Moment_Failures_1")
            Set_Moment_Failures_1.setValues(MOMENTS_1)
        
            Set_Moment_Failures_2 = ampl.getSet("Set_Moment_Failures_2")
            Set_Moment_Failures_2.setValues(MOMENTS_2)
        
            Set_Reference_Failures_1 = ampl.getSet("Set_Reference_Failures_1")
            Set_Reference_Failures_1.setValues(REFERENCE_MOMENTS_1)
        
            Set_Reference_Failures_2 = ampl.getSet("Set_Reference_Failures_2")
            Set_Reference_Failures_2.setValues(REFERENCE_MOMENTS_2)
        
            val = ampl.getParameter("val")
            val.set(value)

            tt = ampl.getParameter("tt")
            tt.set(time_period)
            
            v = ampl.getParameter("v")
            v.set(variation)

            ampl.read('ProactiveOnline_PMLALG_F2.txt')

            retrain = ampl.getParameter("retrain")
            subp = ampl.getParameter("subp")
            counter_retrain = ampl.getParameter("counter_retrain")

            retrain = retrain.value()
            subp = subp.value()
            counter_retrain = counter_retrain.value()
            
            while retrain == 1:
                if value == 1 and variation == 1:
                    x_input_pred_1 = list()
                    x_input_pred_2 = list()

                    if time_period == 1:
                        previous_real_failures_1 = read_csv('DATASET_MACHINE_1_FEATURES.csv', header = 0)
                        previous_real_failures_2 = read_csv('DATASET_MACHINE_2_FEATURES.csv', header = 0)
                    else:
                        previous_real_failures_1 = read_csv('DATASET_MACHINE_1_FEATURES_1.csv', header = 0)
                        previous_real_failures_2 = read_csv('DATASET_MACHINE_2_FEATURES_1.csv', header = 0)
                    
                    del previous_real_failures_1['Period']
                    del previous_real_failures_2['Period']

                    previous_real_failures_1 = array(previous_real_failures_1)
                    previous_real_failures_2 = array(previous_real_failures_2)

                    for i in range(len(previous_real_failures_1) - (112 - int(subp)), len(previous_real_failures_1)):
                        x_input_pred_1.append(previous_real_failures_1[i])

                    for i in range(len(previous_real_failures_2) - (112 - int(subp)), len(previous_real_failures_2)):
                        x_input_pred_2.append(previous_real_failures_2[i])

                    if time_period == 1:
                        real_failures_1 = read_csv('DATASET_MACHINE_1_FEATURES_1.csv', header = 0)
                        real_failures_2 = read_csv('DATASET_MACHINE_2_FEATURES_1.csv', header = 0)
                    else:
                        real_failures_1 = read_csv('DATASET_MACHINE_1_FEATURES_2.csv', header = 0)
                        real_failures_2 = read_csv('DATASET_MACHINE_2_FEATURES_2.csv', header = 0)
                    
                    del real_failures_1['Period']
                    del real_failures_2['Period']

                    real_failures_1 = array(real_failures_1)
                    real_failures_2 = array(real_failures_2)

                    for i in range(0,int(subp)):
                        x_input_pred_1.append(real_failures_1[i])
                        x_input_pred_2.append(real_failures_2[i])

                    x_input_pred_1 = array(x_input_pred_1)
                    x_input_pred_2 = array(x_input_pred_2)

                    x_input_pred_1 = x_input_pred_1.reshape((1,112, 5))    
                    x_input_pred_2 = x_input_pred_2.reshape((1,112, 5)) 

                    n_future_pred = 112 - int(subp)

                    x_input_pred_1 = np.asarray(x_input_pred_1).astype(np.float32)
                    x_input_pred_2 = np.asarray(x_input_pred_2).astype(np.float32)

                    for m in range(1,3):
                        if m == 1:
                            model = keras.models.load_model('final_forecasting_M1')
                            model.load_weights("weights_forecasting_M1.h5")
                            newmodel = tf.keras.Sequential()
                            newmodel.add(tf.keras.layers.LSTM(32, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, batch_input_shape= (1, None, 5), return_sequences = True))
                            newmodel.add(tf.keras.layers.LeakyReLU(alpha = 0.5)) 
                            newmodel.add(tf.keras.layers.LSTM(32, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, return_sequences = True))
                            newmodel.add(tf.keras.layers.LeakyReLU(alpha = 0.5)) 
                            newmodel.add(tf.keras.layers.LSTM(32, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, return_sequences = True))
                            newmodel.add(tf.keras.layers.LSTM(32, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, return_sequences = False))
                            newmodel.add(tf.keras.layers.Dense(5))

                            x_input_pred_II = x_input_pred_1
                        else:
                            model = keras.models.load_model('final_forecasting_M2')
                            model.load_weights("weights_forecasting_M2.h5")
                            newmodel = tf.keras.Sequential()
                            newmodel.add(tf.keras.layers.LSTM(64, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, batch_input_shape= (1, None, 5), return_sequences = True))
                            newmodel.add(tf.keras.layers.LSTM(64, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, return_sequences = True))
                            newmodel.add(tf.keras.layers.LSTM(64, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, return_sequences = True))
                            newmodel.add(tf.keras.layers.LSTM(64, kernel_regularizer = regularizers.l1(l1 = 0.00001), stateful = True, return_sequences = False))
                            newmodel.add(tf.keras.layers.Dense(5))

                            x_input_pred_II = x_input_pred_2

                        newmodel.set_weights(model.get_weights())   

                        x_input_pred_II = array(x_input_pred_II)
                        x_input_pred_II = np.reshape(x_input_pred_II, (112, 5))  

                        if m == 1:
                            dataset = read_csv('DATASET_MACHINE_1_FEATURES.csv', header = 0)
                        else:
                            dataset = read_csv('DATASET_MACHINE_2_FEATURES.csv', header = 0)

                        del dataset['Period']

                        train = dataset[0:8602]

                        train = np.asarray(train)

                        def scale(train, data):
                            # fit scaler
                            scaler = MinMaxScaler()
                            train = train.astype(float)
                            data = data.astype(float)
                            scaler = scaler.fit(train)
                            
                            # transform data
                            data_scaled = scaler.transform(data)
                            return scaler, data_scaled

                        train_Pressure = train[:,0]
                        train_Pressure = train_Pressure.reshape(-1, 1)

                        data_Pressure = x_input_pred_II[:,0]
                        data_Pressure = data_Pressure.reshape(-1, 1)

                        scaler_Pressure, data_Pressure = scale(train_Pressure, data_Pressure)

                        #---

                        train_Speed = train[:,1]
                        train_Speed = train_Speed.reshape(-1, 1)

                        data_Speed = x_input_pred_II[:,1]
                        data_Speed = data_Speed.reshape(-1, 1)

                        scaler_Speed, data_Speed = scale(train_Speed, data_Speed)

                        #---

                        train_Temperature = train[:,2]
                        train_Temperature = train_Temperature.reshape(-1, 1)

                        data_Temperature = x_input_pred_II[:,2]
                        data_Temperature = data_Temperature.reshape(-1, 1)

                        scaler_Temperature, data_Temperature = scale(train_Temperature, data_Temperature)

                        #---

                        train_Sound = train[:,3]
                        train_Sound = train_Sound.reshape(-1, 1)

                        data_Sound = x_input_pred_II[:,3]
                        data_Sound = data_Sound.reshape(-1, 1)

                        scaler_Sound, data_Sound = scale(train_Sound, data_Sound)

                        #---

                        train_Vibration = train[:,4]
                        train_Vibration = train_Vibration.reshape(-1, 1)

                        data_Vibration = x_input_pred_II[:,4]
                        data_Vibration = data_Vibration.reshape(-1, 1)

                        scaler_Vibration, data_Vibration = scale(train_Vibration, data_Vibration)
                        
                        in_seq1 = array(data_Pressure)
                        in_seq2 = array(data_Speed)
                        in_seq3 = array(data_Temperature)
                        in_seq4 = array(data_Sound)
                        in_seq5 = array(data_Vibration)

                        x_input_pred_II = hstack((in_seq1, in_seq2, in_seq3, in_seq4, in_seq5))

                        x_input_pred_II = array(x_input_pred_II)
                        x_input_pred_II = x_input_pred_II.reshape((1,112, 5))
                        x_input_pred_II = np.asarray(x_input_pred_II).astype(np.float32)

                        predictions = []
                        for i in range(n_future_pred):
                            predictions = np.append(predictions, newmodel(x_input_pred_II))
                            predictions = np.asarray(predictions).reshape((1, 1, 1 + i, 5))
                            x_input_pred_II = np.append(x_input_pred_II, predictions[-1], axis = 1)

                        predictions = array(predictions)
                        predictions = np.reshape(predictions, (n_future_pred, 5))

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
                            results_M1 = open("/Forecastings/Results_M1_%d_%d.txt" % (time_period, counter_retrain),"w") 
                            np.savetxt(results_M1, predictions_Pressure)
                            np.savetxt(results_M1, predictions_Speed)
                            np.savetxt(results_M1, predictions_Temp)
                            np.savetxt(results_M1, predictions_Sound)
                            np.savetxt(results_M1, predictions_Vibration)
                            
                            results_M1.close()
                        else:
                            results_M2 = open("/Forecastings/Results_M2_%d_%d.txt" % (time_period, counter_retrain),"w") 
                            np.savetxt(results_M2, predictions_Pressure)
                            np.savetxt(results_M2, predictions_Speed)
                            np.savetxt(results_M2, predictions_Temp)
                            np.savetxt(results_M2, predictions_Sound)
                            np.savetxt(results_M2, predictions_Vibration)

                            results_M2.close()                      

                        if m == 1:
                            ampl.eval('reset data Pressure, Speed, Temperature, Sound, Vibration;')
                            save_predictions(predictions_Pressure, predictions_Speed, predictions_Temp, predictions_Sound, predictions_Vibration)
                        else:
                            ampl.eval('reset data Pressure_M2, Speed_M2, Temperature_M2, Sound_M2, Vibration_M2;')
                            save_predictions(predictions_Pressure, predictions_Speed, predictions_Temp, predictions_Sound, predictions_Vibration)

                        dataset_class = pd.DataFrame(list(zip(predictions_Pressure, predictions_Speed, predictions_Temp, predictions_Sound, predictions_Vibration)), columns =['Pressure', 'Speed', 'Temperature', 'Sound', 'Vibration'])

                        input_Pressure_class = dataset_class['Pressure'].values
                        input_Pressure_class = input_Pressure_class.reshape(-1, 1)

                        scaler_Pressure_class, input_Pressure_class = scale_data(input_Pressure_class)

                        #---

                        input_Speed_class = dataset_class['Speed'].values
                        input_Speed_class = input_Speed_class.reshape(-1, 1)

                        scaler_Speed_class, input_Speed_class = scale_data(input_Speed_class)

                        #---

                        input_Temperature_class = dataset_class['Temperature'].values
                        input_Temperature_class = input_Temperature_class.reshape(-1, 1)

                        scaler_Temperature_class, input_Temperature_class = scale_data(input_Temperature_class)

                        #---

                        input_Sound_class = dataset_class['Sound'].values
                        input_Sound_class = input_Sound_class.reshape(-1, 1)

                        scaler_Sound_class, input_Sound_class = scale_data(input_Sound_class)

                        #---

                        input_Vibration_class = dataset_class['Vibration'].values
                        input_Vibration_class = input_Vibration_class.reshape(-1, 1)

                        scaler_Vibration_class, input_Vibration_class = scale_data(input_Vibration_class)

                        #---

                        X_class = pd.DataFrame(list(zip(array(input_Pressure_class), array(input_Sound_class), array(input_Speed_class), array(input_Temperature_class), array(input_Vibration_class))), columns =['Pressure', 'Speed', 'Temperature', 'Sound', 'Vibration'])
                        X_class = np.asarray(X_class).astype('float32')

                        if m == 1:
                            classification_model = keras.models.load_model('final_model_M1')
                            classification_model.load_weights("weights_M1.h5")
                        else:
                            classification_model = keras.models.load_model('final_model_M2')
                            classification_model.load_weights("weights_M2.h5")

                        # make probability predictions with the model
                        predictions = classification_model(X_class)

                        if m == 1:
                            results_M1 = open("/Forecastings/Classification_M1_%d_%d.txt" % (time_period, counter_retrain),"w") 
                            np.savetxt(results_M1, predictions)
                            
                            results_M1.close()
                        else:
                            results_M2 = open("/Forecastings/Classification_M2_%d_%d.txt" % (time_period, counter_retrain),"w") 
                            np.savetxt(results_M2, predictions)

                            results_M2.close() 
                            
                        if time_period == 1:
                            if m == 1:
                                pred_M1 = []
                                pred_M1 = [np.round(x[0]) for x in predictions]
                                pred_M1 = np.asarray(pred_M1).astype('float32')

                                pred_M1 = pred_M1.reshape(-1, 1)

                                j = 0
                                for i in pred_M1:
                                    if j + 1 < len(pred_M1):
                                        if pred_M1[j] == 1 and pred_M1[j + 1] == 1:
                                            pred_M1[j] = 0
                                    j = j + 1
                            else:
                                pred_M2 = []
                                pred_M2 = [np.round(x[0]) for x in predictions]
                                pred_M2 = np.asarray(pred_M2).astype('float32')

                                pred_M2 = pred_M2.reshape(-1, 1)

                                j = 0
                                for i in pred_M2:
                                    if j + 1 < len(pred_M2):
                                        if pred_M2[j] == 1 and pred_M2[j + 1] == 1:
                                            pred_M2[j] = 0
                                    j = j + 1    
                        else:
                            if m == 1:
                                pred_M1_2 = []
                                pred_M1_2 = [np.round(x[0]) for x in predictions]
                                pred_M1_2 = np.asarray(pred_M1_2).astype('float32')

                                pred_M1_2 = pred_M1_2.reshape(-1, 1)

                                j = 0
                                for i in pred_M1_2:
                                    if j + 1 < len(pred_M1_2):
                                        if pred_M1_2[j] == 1 and pred_M1_2[j + 1] == 1:
                                            pred_M1_2[j] = 0
                                    j = j + 1
                            else:
                                pred_M2_2 = []
                                pred_M2_2 = [np.round(x[0]) for x in predictions]
                                pred_M2_2 = np.asarray(pred_M2_2).astype('float32')

                                pred_M2_2 = pred_M2_2.reshape(-1, 1)

                                j = 0
                                for i in pred_M2_2:
                                    if j + 1 < len(pred_M2_2):
                                        if pred_M2_2[j] == 1 and pred_M2_2[j + 1] == 1:
                                            pred_M2_2[j] = 0
                                    j = j + 1
                else:
                    for m in range(1,3):
                        if m == 1:
                            results = open("/Forecastings/Results_M1_%d_%d.txt" % (time_period, counter_retrain),"r") 
                        else:
                            results = open("/Forecastings/Results_M2_%d_%d.txt" % (time_period, counter_retrain),"r") 

                        text = results.read().splitlines()
                        i = 0
                        predictions_Pressure = []
                        predictions_Speed = []
                        predictions_Temp = []
                        predictions_Sound = []
                        predictions_Vibration = []

                        number_data = 112 - int(subp)

                        for line in text:
                            if i < number_data:
                                predictions_Pressure = np.append(predictions_Pressure, line)
                            else:
                                if i >= number_data and i < 2*number_data:
                                    predictions_Speed = np.append(predictions_Speed, line)    
                                else:
                                    if i >= 2*number_data and i < 3*number_data:
                                        predictions_Temp = np.append(predictions_Temp, line)    
                                    else:
                                        if i >= 3*number_data and i < 4*number_data:
                                            predictions_Sound = np.append(predictions_Sound, line)        
                                        else:
                                            predictions_Vibration = np.append(predictions_Vibration, line)                   
                            i = i + 1
                        results.close()

                        predictions_Pressure = np.asarray(predictions_Pressure, dtype = 'float64')
                        predictions_Speed = np.asarray(predictions_Speed, dtype = 'float64')
                        predictions_Temp = np.asarray(predictions_Temp, dtype = 'float64')
                        predictions_Sound = np.asarray(predictions_Sound, dtype = 'float64')
                        predictions_Vibration = np.asarray(predictions_Vibration, dtype = 'float64')

                        # More decimals to avoid repeated numbers in the sets of 'save_predictions'
                        predictions_Pressure = np.around(predictions_Pressure, decimals = 6, out = None)
                        predictions_Speed = np.around(predictions_Speed, decimals = 6, out = None)
                        predictions_Temp = np.around(predictions_Temp, decimals = 6, out = None)
                        predictions_Sound = np.around(predictions_Sound, decimals = 6, out = None)
                        predictions_Vibration = np.around(predictions_Vibration, decimals = 6, out = None)

                        if m == 1:
                            ampl.eval('reset data Pressure, Speed, Temperature, Sound, Vibration;')
                            save_predictions(predictions_Pressure, predictions_Speed, predictions_Temp, predictions_Sound, predictions_Vibration)

                            predictions = open("/Forecastings/Classification_M1_%d_%d.txt" % (time_period, counter_retrain),"r") 
                        else:
                            ampl.eval('reset data Pressure_M2, Speed_M2, Temperature_M2, Sound_M2, Vibration_M2;')
                            save_predictions(predictions_Pressure, predictions_Speed, predictions_Temp, predictions_Sound, predictions_Vibration)

                            predictions = open("/Forecastings/Classification_M2_%d_%d.txt" % (time_period, counter_retrain),"r")

                        text = predictions.read().splitlines()
                        predictions = []

                        for line in text:
                            predictions = np.append(predictions, line)

                        predictions = np.asarray(predictions, dtype = 'float64')
                        predictions = predictions.reshape(-1, 1)
                            
                        if time_period == 1:
                            if m == 1:
                                pred_M1 = []
                                pred_M1 = [np.round(x[0]) for x in predictions]
                                pred_M1 = np.asarray(pred_M1).astype('float32')

                                pred_M1 = pred_M1.reshape(-1, 1)

                                j = 0
                                for i in pred_M1:
                                    if j + 1 < len(pred_M1):
                                        if pred_M1[j] == 1 and pred_M1[j + 1] == 1:
                                            pred_M1[j] = 0
                                    j = j + 1
                            else:
                                pred_M2 = []
                                pred_M2 = [np.round(x[0]) for x in predictions]
                                pred_M2 = np.asarray(pred_M2).astype('float32')

                                pred_M2 = pred_M2.reshape(-1, 1)

                                j = 0
                                for i in pred_M2:
                                    if j + 1 < len(pred_M2):
                                        if pred_M2[j] == 1 and pred_M2[j + 1] == 1:
                                            pred_M2[j] = 0
                                    j = j + 1    
                        else:
                            if m == 1:
                                pred_M1_2 = []
                                pred_M1_2 = [np.round(x[0]) for x in predictions]
                                pred_M1_2 = np.asarray(pred_M1_2).astype('float32')

                                pred_M1_2 = pred_M1_2.reshape(-1, 1)

                                j = 0
                                for i in pred_M1_2:
                                    if j + 1 < len(pred_M1_2):
                                        if pred_M1_2[j] == 1 and pred_M1_2[j + 1] == 1:
                                            pred_M1_2[j] = 0
                                    j = j + 1
                            else:
                                pred_M2_2 = []
                                pred_M2_2 = [np.round(x[0]) for x in predictions]
                                pred_M2_2 = np.asarray(pred_M2_2).astype('float32')

                                pred_M2_2 = pred_M2_2.reshape(-1, 1)

                                j = 0
                                for i in pred_M2_2:
                                    if j + 1 < len(pred_M2_2):
                                        if pred_M2_2[j] == 1 and pred_M2_2[j + 1] == 1:
                                            pred_M2_2[j] = 0
                                    j = j + 1

                k = 1
                j = 1

                MOMENTS_1 = list()
                MOMENTS_2 = list()

                if time_period == 1:
                    for period in range(112 - int(subp)):
                        if pred_M1[period] == 1:
                            MOMENTS_1.append(int(subp) + period + 1)  
                        
                        if pred_M2[period] == 1:
                            MOMENTS_2.append(int(subp) + period + 1)
                else:
                    for period in range(112 - int(subp)):
                        if pred_M1_2[period] == 1:
                            MOMENTS_1.append(int(subp) + period + 1)
                        
                        if pred_M2_2[period] == 1:
                            MOMENTS_2.append(int(subp) + period + 1)

                ampl.eval('reset data Set_Moment_Failures_1, Set_Moment_Failures_2;')
                Set_Moment_Failures_1 = ampl.getSet("Set_Moment_Failures_1")
                Set_Moment_Failures_1.setValues(MOMENTS_1)
            
                Set_Moment_Failures_2 = ampl.getSet("Set_Moment_Failures_2")
                Set_Moment_Failures_2.setValues(MOMENTS_2)

                ampl.read('ProactiveOnline_PMLALG_F2.txt')

                retrain = ampl.getParameter("retrain")
                subp = ampl.getParameter("subp")
                counter_retrain = ampl.getParameter("counter_retrain")

                retrain = retrain.value()
                subp = subp.value()
                counter_retrain = counter_retrain.value()

        fim = time.time()
        time_measure = fim - ini
        time_measure = np.array([time_measure])
        times = open("Time_PMLALG_F2.txt","a") 
        np.savetxt(times, time_measure)
        times.close() 