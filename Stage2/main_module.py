# -*- coding: utf-8 -*-
"""
Created on Fri Aug 2 17:01:54 2024

@author: abhishek
"""

import argparse as aP
import numpy as np
import time
import lp_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import wnch
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import lap_score
import construct_W
import pandas as pd
k = float('-inf')

	
if __name__=='__main__':
  start_time=time.time()
  X=pd.read_csv('../pc1.csv',header = None).to_numpy()
  n_samples,n_feature=X.shape
  data=X[:,0:n_feature]
  object= StandardScaler()
  datav = object.fit_transform(data)
  kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
  W = construct_W.construct_W(datav, **kwargs_W)
  score = lap_score.lap_score(datav, W=W)
  print(score)
  M=sorted(score)
  print(M)
  b = lap_score.feature_ranking(score)
  print (b)
  s=np.array([])
  for i in range(n_feature):
   s = np.union1d(s, b[i]) 
   print (s)
   wnch.wnch(list(s.astype(np.int32)), M[i], i)
  print("--- %s seconds ---" % (time.time() - start_time))