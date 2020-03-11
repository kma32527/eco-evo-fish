# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 15:37:09 2019

@author: Kevin
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import random

data=raw_examples[1:]
random.shuffle(data)
train=data[0:150]
val=data[150:300]
test=data[300:]

traintargets=[entry[len(entry)-1] for entry in train]
valtargets=[entry[len(entry)-1] for entry in val]
traindata=[entry[:len(entry)-1] for entry in train]
valdata=[entry[:len(entry)-1] for entry in val]

model=LogisticRegression(penalty='none', random_state=0, solver='liblinear')

model.fit(traindata, traintargets)
print(model.score(valdata, valtargets))