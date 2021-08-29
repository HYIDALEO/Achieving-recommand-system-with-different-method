import csv
import math
from collections import defaultdict
from surprise import SVD
from surprise import Dataset, Reader
import os
# from sklearn.decomposition import NMF
import numpy as np
import pandas as pd
from surprise import NMF
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt




def sur_NMF():
    file_path = os.path.expanduser('train50.csv')
    reader = Reader(line_format='user item rating timestamp', sep=',')
    data = Dataset.load_from_file(file_path, reader)
    trainset = data.build_full_trainset()
    res = []
    res_x  = []
    for x in range(50,251,5):
        nmf = NMF(n_factors = 200,n_epochs =80,reg_pu =0.07,reg_qi =0.06)
        #print(trainset)
        nmf.fit(trainset)
        f = open('test50.csv', 'r')
        true = []
        predic = []
        for line in f:
            line = line.strip().split(',')
            uid = line[0]
            movieId = line[1]
            #temp = nmf.predict(uid, movieId,line[2],line[3],line[4],line[5],line[6],line[7],line[8],line[9],line[10],line[11],line[12],line[13],line[14],line[15],line[16],line[17],line[18],line[19],line[20],line[21])
            temp = nmf.predict(uid, movieId)
            true.append(float(line[2]))
            predic.append(temp.est)
        f.close()
        rmse = np.sqrt(mean_squared_error(true, predic))
        res.append(rmse)
        res_x.append(x)
        print(rmse,x)

    plt.plot(res_x, res, marker='o', mec='r', mfc='w',label=u'')
    plt.legend()
    plt.xlabel(u"n_factors") #X轴标签
    plt.ylabel("RMSE") #Y轴标签
    plt.title("n_epochs = 80") #标题

    plt.show()


def createNewSet():
    res = []
    train_res = []
    test_res = []
    movies_info = []
    with open('movies.csv', 'r') as csvfile:
        reader_ = csv.DictReader(csvfile)
        for row_ in reader_:
            fitures = row_['genres'].split('|')
            for ele in fitures:
                if ele not in res:
                    res.append(ele)
    for ele in res:
        print(ele)
    with open('movies.csv', 'r') as csvfile:
        reader_ = csv.DictReader(csvfile)
        for row_ in reader_:
            fitures = row_['genres'].split('|')
            movie_info = []
            movie_info.append(int(row_['movieId']))
            # len = 20
            for index_ele in range(0,len(res)):
                if res[index_ele] in fitures:
                    movie_info.append(1)
                else:
                    movie_info.append(0)
            movies_info.append(movie_info)

    for ele in movies_info:
        print(ele)


    f_train = open('train50.csv', 'r')
    for line in f_train:
        line = line.strip().split(',')
        train_r = []
        train_r.append(line[0])
        train_r.append(line[1])
        for ele in movies_info:
            if ele[0] == int(line[1]):
                for index_ele in range(1,len(ele)):
                    train_r.append(ele[index_ele])
                break
        train_r.append(line[2])
        train_r.append(line[3])
        train_res.append(train_r)

    f = open('train50withFiture.csv', 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)
    row_list = ["userId", "movieId"]
    for index_ele in range(0, len(res)):
        row_list.append(res[index_ele])
    row_list.append("rating")
    row_list.append("timestamp")
    csv_writer.writerow(row_list)
    for train_ele in train_res:
        csv_writer.writerow(train_ele)
    f.close()

#    f_test = open('test50.csv', 'r')

#createNewSet()
sur_NMF()