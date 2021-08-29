from random import shuffle
import numpy as np
import pandas as pd
import csv

df = pd.read_csv('C://Users/admin/Desktop/semB/6002/ml-latest-small/ratings.csv')
predictors = ['userId','movieId','rating','timestamp']

trainset = df.values
ran_test = []
ran_train = []
for i in range(len(trainset)):
    trainset[i][0] = int(trainset[i][0])
    trainset[i][1] = int(trainset[i][1])


def randomDivide(percent):
    last_index = 0
    count_with_same_user = 0
    for i in range(0,len(trainset)):
        if i+1 < len(trainset) and int(trainset[i][0]) == int(trainset[i+1][0]):
            count_with_same_user+=1
        elif (i+1 < len(trainset) and int(trainset[i][0]) != int(trainset[i+1][0])) or i+1 == len(trainset):
            count_with_same_user += 1
            # range(10) =>  0~9
            index_arr = [index for index in range(count_with_same_user)]
            shuffle(index_arr)
            mark_index = 0
            for count_index in range(int(count_with_same_user * percent)):
                ran_test.append(trainset[last_index + index_arr[count_index]])
                mark_index = count_index

            for count_index in range(mark_index+1,count_with_same_user):
                ran_train.append(trainset[last_index + index_arr[count_index]])
            last_index += count_with_same_user
            count_with_same_user = 0


    return 0
randomDivide(0.5)
print(ran_test[0])
print(ran_train[0])

f = open('test50.csv','w',encoding='utf-8',newline="")
csv_writer = csv.writer(f)
csv_writer.writerow(["userId","movieId","rating","timestamp"])
for test_ele in ran_test:
    csv_writer.writerow(test_ele)

f.close()

f = open('train50.csv','w',encoding='utf-8',newline="")
csv_writer = csv.writer(f)
csv_writer.writerow(["userId","movieId","rating","timestamp"])
for train_ele in ran_train:
    csv_writer.writerow(train_ele)

f.close()