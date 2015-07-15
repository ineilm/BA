# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 21:25:53 2015
@author: Indranil
"""

from sklearn.externals import joblib
import pandas as pad
from sklearn.preprocessing import MinMaxScaler


predict_data = pad.read_csv("Enter your csv file here for predicting new category",header=None)
clf =MinMaxScaler()
data_to_predict = clf.fit_transform(predict_data)
best_fit_model = joblib.load("Model.pkl")
predictions = best_fit_model.predict(data_to_predict)
for vals in predictions:
    for points in predict_data:
        print "Given data points :"+str(points)
        print "Predicted Value"+str(vals)





