import csv
import math
from collections import defaultdict
from surprise import SVD

from surprise import Dataset, Reader
import os
from sklearn.decomposition import NMF
import numpy as np
import pandas as pd

train_path = 'C://Users/admin/Desktop/semB/6002/ml-latest-small/train50.csv'
test_path = 'C://Users/admin/Desktop/semB/6002/ml-latest-small/test50.csv'
predictors = ['userId','movieId','rating','timestamp']
#print(df)


nmf_model = NMF(n_components=2) # 设有2个主题
def getMatrix(path):
    df = pd.read_csv(path)
    trainset = df.values
    RATE_MATRIX = []
    us = []
    # 0 mv 1 ra
    for x in trainset:
        if int(x[0]) not in us:
            us.append(int(x[0]))
            RATE_MATRIX.append([])

    mv = []
    for x in trainset:
        if int(x[1]) not in mv:
            mv.append(int(x[1]))
    print(len(mv))
    for x in us:
        for y in mv:
            RATE_MATRIX[us.index(x)].append(0)

    for x in trainset:
        RATE_MATRIX[us.index(int(x[0]))][mv.index(int(x[1]))] = x[2]
    return RATE_MATRIX, us, mv


train_mat,us,mv = getMatrix(train_path)
test_mat,test_us,test_mv = getMatrix(test_path)
item_dis = nmf_model.fit_transform(train_mat)
user_dis = nmf_model.components_

# print('用户的主题分布：')
# print(user_dis)
# print('电影的主题分布：')
# print(item_dis)

filter_matrix  = train_mat
# print(RATE_MATRIX)
rec_mat = np.dot(item_dis, user_dis)
# print('重建矩阵，并过滤掉已经评分的物品：')
rec_filter_mat = (filter_matrix * rec_mat).T
# print(rec_filter_mat)

rec_user = '凛冬将至'  # 需要进行推荐的用户
rec_userid = 1  # 推荐用户ID
rec_list = rec_filter_mat[us.index(rec_userid), :]  # 推荐用户的电影列表

#print('推荐用户的电影：')
#print(np.nonzero(rec_list))

#print(b.reconstruction_err_)

# c = NMF(n_components=4)  # 设有4个主题
# W = c.fit_transform(RATE_MATRIX)
# H = c.components_
#print(c.reconstruction_err_)

# d = NMF(n_components=100)  # 设有5个主题
# W = d.fit_transform(RATE_MATRIX)
# H = d.components_
#print(d.reconstruction_err_)
def matrix_factorization_predict(user_lst, item_lst, rate_lst, matrix_w, matrix_h):
    max_user_id = len(matrix_w)
    max_item_id = len(matrix_h[0])
    count = len(user_lst)
    error = 0
    for user, item, rate in zip(user_lst, item_lst, rate_lst):
        # filter the triple whose user or item are not trained.
        if user >= max_user_id or item > max_item_id:
            continue
        error += abs(rate - np.dot(matrix_w[user, :], matrix_h[:, item]))
    rmse = math.sqrt(error / count)
    return rmse
# f = open('t1.csv','w',encoding='utf-8',newline="")
# csv_writer = csv.writer(f)
# for test_ele in test_mat:
#     csv_writer.writerow(test_ele)

def getfileMat(inp_set):
    res = []



n_comps = [1,2,3,4,5,6,8,10,13,15,17,20,30,40,50]
alphas = [0.01,0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.7]
for n_comp in n_comps:
    for alp in alphas:
        nmf = NMF(n_components=3)
        item_dis = nmf.fit_transform(train_mat)
        user_dis = nmf.components_

        rec_mat = np.dot(item_dis, user_dis)

        for u_index in range(len(us)):
            for m_index in range(len(mv)):
                u_value = us[u_index]
                m_value = mv[m_index]
                m_rate = rec_mat[u_index][m_index]
                f = open('t2.csv','w',encoding='utf-8',newline="")
                csv_writer = csv.writer(f)
                for test_ele in rec_mat:
                    csv_writer.writerow(test_ele)
                print(rec_mat.shape)
                print(us[u_index],u_index)
                print(mv[m_index],m_index)

                t_rate = train_mat[u_index][m_index]
                print(m_rate,t_rate)
                break
            break
        break
    break
