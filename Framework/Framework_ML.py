# Machine learning framework - Choose the best strategy to be implemented (proactive, proactive-online, or corrective)

from Cluster import *
from Optimization import *
from ML import *
from pandas import read_csv
import pandas as pd
import random
import numpy as np
import tensorflow as tf
from numpy import array
from numpy import hstack
from scipy import stats

totalStrat = 13

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

datafile = "Datafile.txt"
    
n = Clustering(datafile)

Optimization(totalStrat, n)

model = MachineLearning()

def Read_Data(filename):
    data = []
    with open(filename) as f:
        content = f.read().splitlines()
        for line in content:
            contentBreak = line.split()
            for temp in contentBreak:
                data.append(temp)
    return data

for value in range(1,10):
    for variation in range(1,16):
        value = int(float(value))
        variation = int(float(variation))

        if value == 1:
            value = 4
        else:
            if value == 2:
                value = 6
            else:
                if value == 3:
                    value = 8
                else:
                    if value == 4:
                        value = 10
                    else:
                        if value == 5:
                            value = 12
                        else:
                            if value == 6:
                                value = 15
                            else:
                                if value == 7:
                                    value = 20
                                else:
                                    if value == 8:
                                        value = 50
                                    else:
                                        value = 100


        filename = "Data_Tardiness/p" + str(value) + str(variation) + ".txt"
        p = Read_Data(filename)
        filename = "Data_Tardiness/D" + str(value) + str(variation) + ".txt"
        D = Read_Data(filename)
        filename = "Data_Tardiness/setup" + str(value) + str(variation) + ".txt"
        setup = Read_Data(filename)

        p = np.array(p).astype(np.float)
        D = np.array(D).astype(np.float)
        setup = np.array(setup).astype(np.float)

        Average_p = np.sum(p, axis = 0)/value
        Average_D = np.sum(D, axis = 0)/(value*2) # Two periods of the planning horizon
        Average_setup = np.sum(setup, axis = 0)/(value*value + value)

        DataN =[np.array([value, Average_p, Average_D, Average_setup]).tolist()]
        Base = np.genfromtxt("Calibracao.txt", delimiter = ',', usecols = range(0,5)).astype(float).tolist()   
        Base = np.delete(Base, (1), axis = 1)     

        TestStat = stats.ttest_ind(DataN, Base, 1)[1]

        strat = []
        strat.append('/Strategies/Script_PMLINST_WF.py') # PMLINST WF
        strat.append('/Strategies/Script_PMLINST_F1.py') # PMLINST F1
        strat.append('/Strategies/Script_PMLINST_F2.py') # PMLINST F2
        strat.append('/Strategies/Script_PMLALG_WF.py') # PMLALG WF
        strat.append('/Strategies/Script_PMLALG_F1.py') # PMLINST F1
        strat.append('/Strategies/Script_PMLALG_F2.py') # PMLINST F2        
        strat.append('/Strategies/Script_ProactiveOnline_PMLINST_WithoutFeedback - II.py') # PMLINST WF ProactiveOnline
        strat.append('/Strategies/Script_ProactiveOnline_PMLINST_F1 - II.py') # PMLINST F1 ProactiveOnline
        strat.append('/Strategies/Script_ProactiveOnline_PMLINST_F2 - II.py') # PMLINST F2 ProactiveOnline
        strat.append('/Strategies/Script_ProactiveOnline_PMLALG_WithoutFeedback - II.py') # PMLALG WF ProactiveOnline
        strat.append('/Strategies/Script_ProactiveOnline_PMLALG_F1 - II.py') # PMLINST F1 ProactiveOnline
        strat.append('/Strategies/Script_ProactiveOnline_PMLALG_F2 - II.py') # PMLINST F2 ProactiveOnline
        strat.append('/Strategies/Script_Corrective.py') # Corrective

        # If the two samples have different means, we reject the null hypothesis and retrain the model  
        if np.max(TestStat) < 0.05:
            Rest = open("/Calibracao.txt","a+")

            Best = Infinity

            value = int(float(value))
            variation = int(float(variation))

            if value == 4:
                value = 1
            else:
                if value == 6:
                    value = 2
                else:
                    if value == 8:
                        value = 3
                    else:
                        if value == 10:
                            value = 4
                        else:
                            if value == 12:
                                value = 5
                            else:
                                if value == 15:
                                    value = 6
                                else:
                                    if value == 20:
                                        value = 7
                                    else:
                                        if value == 50:
                                            value = 8
                                        else:
                                            value = 9

            for i in range(0, totalStrat):
                from runpy import run_path
                result = run_path(strat[i], init_globals={"value": value, "variation": variation})
                result = result.get('tardiness')
    
                if result < Best:
                    Best = result
                    Best_Strategy = i        

            Rest.write("%s,"%value)
            Rest.write("%s,"%variation)
            Rest.write("%s,"%Average_p)
            Rest.write("%s,"%Average_D)
            Rest.write("%s,"%Average_setup)
            Rest.write("%s,"%str(round(Best,2)))
            Rest.write("%s\n"%str(round(Best_Strategy,2)))
            Rest.close()

            model = MachineLearning()

        x = np.array([[value, Average_p, Average_D, Average_setup]])
        prediction = model.predict(x)

        if value == 4:
            value = 1
        else:
            if value == 6:
                value = 2
            else:
                if value == 8:
                    value = 3
                else:
                    if value == 10:
                        value = 4
                    else:
                        if value == 12:
                            value = 5
                        else:
                            if value == 15:
                                value = 6
                            else:
                                if value == 20:
                                    value = 7
                                else:
                                    if value == 50:
                                        value = 8
                                    else:
                                        value = 9

        #from runpy import run_path
        #result = run_path(strat[int(prediction)], init_globals={"value": value, "variation": variation})
        #result = result.get('tardiness')

        Result = open("/Results.txt","a+")
        Result.write("%s,"%str(value))
        Result.write("%s,"%str(variation))
        #Result.write("%s,"%str(result))
        Result.write("%s\n"%str(prediction[0]))
        Result.close()