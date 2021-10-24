import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

def MachineLearning():
    X = np.genfromtxt("Calibracao.txt", delimiter = ',', usecols = range(0,5)).astype(float) 
    X = np.delete(X, (1), axis = 1)  

    Y = np.genfromtxt("Calibracao.txt", delimiter = ',', usecols = (6)).astype(float) 
    Y = Y.reshape((-1, 1)).astype(float) 

    model = RandomForestClassifier(n_estimators = 100)

    # Using oversampling because some strategies are represented by only one cluster
    steps = [('over', RandomOverSampler()), ('model', model)]
    pipeline = Pipeline(steps = steps)
    classifier = pipeline.fit(X, Y)

    return classifier
    