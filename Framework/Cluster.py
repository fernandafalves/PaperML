from os import replace
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
import random
import numpy as np

def calculate_wcss(data,range_n_clusters):
    wcss = []
    for n in range_n_clusters:
        kmeans = KMeans(n_clusters = n)
        kmeans.fit(X = data)
        wcss.append(kmeans.inertia_)
    return wcss

def optimal_number_of_clusters(wcss,Dim,ClusterMax):
    x1, y1 = Dim, wcss[0]
    x2, y2 = ClusterMax - 1, wcss[len(wcss) - 1]

    distances = []
    for i in range(len(wcss)):
        x0 = i + 1
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 1

def Clustering (Files):
    arquivo = Files
    dfDados = pd.read_csv(arquivo)
    dfDados2 = dfDados
    dfDados = dfDados.iloc[:, 1:]

    Dim = dfDados.shape[1] #Number of variables of the data set
    DimN = dfDados.shape[0]

    ClusterMax = round(0.4*DimN)

    print("ClusterMax:", ClusterMax)
            
    range_n_clusters = list(range(Dim, Dim + ClusterMax))   
        
    # calculating the within clusters sum-of-squares for the cluster amounts
    sum_of_squares = calculate_wcss(dfDados, range_n_clusters)
    
    # calculating the optimal number of clusters
    n = optimal_number_of_clusters(sum_of_squares, Dim, ClusterMax)
    
    print("Optimal:", n)
    
    #plt.plot(range_n_clusters, sum_of_squares, 'bx-')
    #plt.xlabel('k')
    #plt.ylabel('Sum_of_squared_distances')
    #plt.title('Elbow Method For Optimal k')
    #plt.show()
    
    # running kmeans to our optimal number of clusters
    kmeans = KMeans(n_clusters = n, random_state = 0).fit(dfDados)    
    labels = kmeans.predict(dfDados)
    
    dfDados2.insert(Dim, "Cluster", labels, True)
    dfDados2.to_csv("DatasetK.csv")   
    df = pd.read_csv("DatasetK.csv")
    
    grouped_df = df.groupby('Cluster').apply(lambda obj: obj.loc[np.random.choice(obj.index, 5, replace = False),:])    
    keys, values = zip(*grouped_df.items()) # Split values

    Instance_n = values[2]
    Instance_v = values[1]
    Instance_p = values[3]
    Instance_D = values[4]
    Instance_St = values[6]

    Instance_n = Instance_n.values.tolist()
    Instance_v = Instance_v.values.tolist()
    Instance_p = Instance_p.values.tolist()
    Instance_D = Instance_D.values.tolist()
    Instance_St = Instance_St.values.tolist()

    Rest = open("Instances.txt","w+")
    for i in range(len(Instance_n)):
        Rest.write("%s,"%Instance_n[i])
        Rest.write("%s"%Instance_v[i])
        Rest.write("\n")

    Rest2 = open("Instances2.txt","w+")
    for i in range(len(Instance_n)):
        Rest2.write("%s,"%Instance_n[i])
        Rest2.write("%s,"%Instance_v[i])
        Rest2.write("%s,"%Instance_p[i])
        Rest2.write("%s,"%Instance_D[i])
        Rest2.write("%s"%Instance_St[i])
        Rest2.write("\n")
    n = n*5
    return n