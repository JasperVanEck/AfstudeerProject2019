# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:54:57 2019

@author: Jasper van Eck
"""

from sklearn import linear_model as linearRegression
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

hasModelSKRun = False
hasModelSMRun = False

modelSK = 0
modeSM = 0

def trainModel(X, Y):
    global hasModelSKRun
    global modelSK
    if (hasModelSKRun):
        return "Model has run already."
    modelSK = LogisticRegression()
    modelSK.fit(X, Y)
    hasModelSKRun = True
    
    return modelSK

def predict(X):
    return modelSK.predict(X)

def trainModelSM(X, Y):
    global hasModelSMRun
    global modelSM
    if (hasModelSMRun):
        return "Model has run already"
    X = sm.add_constant(X)
    #modelSM = sm.OLS(Y, X).fit()
    modelSM = sm.Logit(Y, X).fit()
    hasModelSMRun = True
    
    return modelSM

def smModelSummary():
    if (hasModelSMRun):
        return modelSM.summary()
    else:
        return "Model has not run yet."