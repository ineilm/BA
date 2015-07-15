# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 00:27:08 2015
@author: Indranil
"""
from sklearn.cross_validation import train_test_split

def data_train_test_split(predictors,target):
    train_predictors,test_predictors,train_targets,test_targets = train_test_split(predictors,target,test_size=0.2)
    return train_predictors,test_predictors,train_targets,test_targets
