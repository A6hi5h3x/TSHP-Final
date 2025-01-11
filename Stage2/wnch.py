# -*- coding: utf-8 -*-
"""
Created on Fri Aug 2 17:01:54 2024

@author: abhishek
"""
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import pandas as pd
from sklearn.metrics import davies_bouldin_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
f = float('-inf')

import numpy as np
from sklearn.cluster import KMeans

sbest = np.array([])
def wnch(s, lr ,t):
 print ("s value",s)
 global f
 global sbest
 global L
 global q
 global dbi
 
 temp = 0
 X = pd.read_csv('../pc1.csv',header = None)
 n_samples,n_feature1= X.shape
 object=StandardScaler()
 V = object.fit_transform(X)
 lrr=1/(lr)
 kmeans_model = KMeans(n_clusters=2, random_state=42).fit(V[:,s])
 labels = kmeans_model.labels_
 Q = metrics.calinski_harabasz_score(V[:,s], labels)
 S = metrics.silhouette_score(V[:,s], labels, metric='euclidean')
 print(Q)
 p= metrics.calinski_harabasz_score(V[:,s], labels)* len(s)*lrr  
 
 if (p>f):
     f = p
     sbest = s
     L = S
     q = Q
     dbi = davies_bouldin_score(V[:,s], labels)

 print(t)
 print(n_feature1)
 if (t+1 == n_feature1):   
    print (f'sbest={sbest},ch={q},dbi={dbi} ,L={L},p={p}')
