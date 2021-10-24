import amplpy
from pandas import read_csv
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
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
from amplpy import AMPL, Environment
ampl = AMPL()

for value in range(1,10):
    for variation in range(1,16):
        SEED = 123
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        for time in range(1,3):
            if time == 1:
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
                    
            if time == 1:
                ampl.eval('reset;')
                ampl.read('Data_PMLINST.txt')


            ampl.eval('reset data Set_Moment_Failures_1, Set_Moment_Failures_2, Set_Reference_Failures_1, Set_Reference_Failures_2;')
        
            Set_Reference_Failures_1 = ampl.getSet("Set_Reference_Failures_1")
            Set_Reference_Failures_1.setValues(REFERENCE_MOMENTS_1)
        
            Set_Reference_Failures_2 = ampl.getSet("Set_Reference_Failures_2")
            Set_Reference_Failures_2.setValues(REFERENCE_MOMENTS_2)
        
            val = ampl.getParameter("val")
            val.set(value)

            tt = ampl.getParameter("tt")
            tt.set(time)
            
            v = ampl.getParameter("v")
            v.set(variation)

            for mach in range(1,3):	
                MFailure = ampl.getParameter("MFailure")
                MFailure.set(mach)
            
                ampl.read('PMLINST.txt')            