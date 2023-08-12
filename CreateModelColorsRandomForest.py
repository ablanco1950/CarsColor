# -*- coding: utf-8 -*-
"""
Created on augost 2023

@author: Alfonso Blanco
"""
#######################################################################
# PARAMETERS
######################################################################

import time
#import cv2
import joblib
Ini=time.time()

import numpy as np
#import os
#import re

###########################################################
# MAIN
##########################################################
from sklearn.ensemble import RandomForestClassifier
arr=[]
arry=[]
arrname=[]
f=open("colors.csv","r")

Conta=0;
for linea in f:
    Conta=Conta+1
    
    lineadelTrain =linea.split(",")
  
 
    linea_x =[]
    z=2
    for x in lineadelTrain:
   
        z=z+1
        if z==6: break
        
        linea_x.append(int(lineadelTrain[z]))
  
    arr.append(linea_x)
    arry.append(int(Conta))
    arrname.append(lineadelTrain[1])
   

X_train=np.array(arr)
#   print(x)
Y_train=np.array(arry)
Name=np.array(arrname)
 
rf= RandomForestClassifier()

modelRf= rf.fit(X_train,Y_train)

joblib.dump(modelRf, "colors_random_forest.joblib")
#modelRf=joblib.load("colors_random_forest.joblib")

