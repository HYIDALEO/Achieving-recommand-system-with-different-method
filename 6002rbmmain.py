import sys

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scrapbook as sb
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages
from sklearn.metrics import mean_squared_error
from rbm import RBM
from python_splitters import numpy_stratified_split
from sparse import AffinityMatrix

data = pd.read_csv(
            'ratings.csv',
            engine="python",
            names=['userID','movieID','rating','timestamp'],
            usecols=[*range(4)]
        )

data.loc[:, 'rating'] = data['rating'].astype(np.int32)

header = {
        "col_user": "userID",
        "col_item": "movieID",
        "col_rating": "rating",
    }

#instantiate the sparse matrix generation
am = AffinityMatrix(df = data, **header)

#obtain the sparse matrix
X, _, _ = am.gen_affinity_matrix()
Xtr, Xtst = numpy_stratified_split(X)

model = RBM(hidden_units= 600, training_epoch = 30, minibatch_size= 60, keep_prob=0.9,with_metrics =True)

train_time= model.fit(Xtr, Xtst)