# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 00:27:41 2015
@author: Indranil
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import pylab as pl
#Reason for using all this classifiers: 
#Incremental Learning : http://scikit-learn.org/stable/modules/scaling_strategies.html



def MultinomialNB_1(train_predictors,test_predictors,train_target,test_target):
    clf = MultinomialNB()
    clf.fit(train_predictors,train_target)
    predicted = clf.predict(test_predictors)
    accuracy = accuracy_score(test_target, predicted)
    print "Accuracy for MultiNomial Naive Bayes: "+str(accuracy)
    return accuracy,predicted  
          
def BernoulliNB_1(train_predictors,test_predictors,train_target,test_target):
    clf = BernoulliNB()
    clf.fit(train_predictors,train_target)
    predicted = clf.predict(test_predictors)
    accuracy = accuracy_score(test_target, predicted)
    print "Accuracy for Bernoulli Naive Bayes: "+str(accuracy)
    return accuracy,predicted  
    
def Perceptron_1(train_predictors,test_predictors,train_target,test_target):
    clf = Perceptron()
    clf.fit(train_predictors,train_target)
    predicted = clf.predict(test_predictors)
    accuracy = accuracy_score(test_target, predicted)
    print "Accuracy for Linear Model Perceptron: "+str(accuracy)
    return accuracy,predicted  
    
def PassiveAggressiveClassifier_1(train_predictors,test_predictors,train_target,test_target):
    clf = PassiveAggressiveClassifier()
    clf.fit(train_predictors,train_target)
    predicted = clf.predict(test_predictors)
    accuracy = accuracy_score(test_target, predicted)
    print "Accuracy for Linear Model PassiveAggressiveClassifier: "+str(accuracy)
    return accuracy,predicted 
    
def plot_roc(test_target,predicted):
    target_names = ['Background', 'Signal']
    print(classification_report(test_target, predicted, target_names=target_names))
    fpr, tpr, thresholds = roc_curve(test_target, predicted)
    roc_auc = auc(fpr, tpr)
    print "Area under the ROC curve : %f" % roc_auc
    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()
    

def build_model_test(train_predictors,test_predictors,train_target,test_target):
    MultinomialNB_accuracy,predicted_MultinomialNB = MultinomialNB_1(train_predictors,test_predictors,train_target,test_target)
    BernoulliNB_accuracy,predicted_BernoulliNB = BernoulliNB_1(train_predictors,test_predictors,train_target,test_target)
    Perceptron_accuracy,predicted_Perceptron = Perceptron_1(train_predictors,test_predictors,train_target,test_target)
    PassiveAggressiveClassifier_accuracy,predicted_PassiveAggressiveClassifier = PassiveAggressiveClassifier_1(train_predictors,test_predictors,train_target,test_target)
    if MultinomialNB_accuracy > BernoulliNB_accuracy and MultinomialNB_accuracy > Perceptron_accuracy and MultinomialNB_accuracy > PassiveAggressiveClassifier_accuracy:
        best_classifier = "Best Model is MultinomialNB"
        plot_roc(test_target,predicted_MultinomialNB)
    elif BernoulliNB_accuracy > MultinomialNB_accuracy and BernoulliNB_accuracy > Perceptron_accuracy and BernoulliNB_accuracy > PassiveAggressiveClassifier_accuracy:
        best_classifier = "Best Model is BernoulliNB"
        plot_roc(test_target,predicted_BernoulliNB)
    elif Perceptron_accuracy > BernoulliNB_accuracy and Perceptron_accuracy > MultinomialNB_accuracy and Perceptron_accuracy > PassiveAggressiveClassifier_accuracy:
        best_classifier = "Best Model is Perceptron"
        plot_roc(test_target,predicted_Perceptron)
    elif PassiveAggressiveClassifier_accuracy > Perceptron_accuracy and PassiveAggressiveClassifier_accuracy > BernoulliNB_accuracy and PassiveAggressiveClassifier_accuracy > MultinomialNB_accuracy:
        best_classifier = "Best Model is PassiveAggressiveClassifier"
        plot_roc(test_target,predicted_PassiveAggressiveClassifier)
    print best_classifier
    return best_classifier
    

