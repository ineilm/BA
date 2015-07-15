# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 00:09:20 2015
@author: Indranil
"""
# All modules imported
import pandas as pad
from Train_Test_Split import data_train_test_split
from Models import build_model_test
from sklearn.preprocessing import MinMaxScaler

#Reading the data file  
data_read_csv = pad.read_csv("C:\\Users\\HP\\Desktop\\bounty app\\SUSY.csv",header=None)
data_target = data_read_csv[[0]]
clf =MinMaxScaler()
data_predictors = clf.fit_transform(data_read_csv.ix[:,1:])

# Split dataset into train and Test
train_predictors,test_predictors,train_target,test_target = data_train_test_split(data_predictors,data_target)
# Build Models in Train data
best_model = build_model_test(train_predictors,test_predictors,train_target,test_target)
print best_model
# Test Model in Test data



