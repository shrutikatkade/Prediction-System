# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 20:48:02 2019

@author: Urjita Kerkar
"""

import pandas as pd
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv("datasets/Test.csv")

data = dataset.iloc[:49,:-1].values
label = dataset.iloc[:49,-1].values

#numpy.save

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
label = labelencoder.fit_transform(label)


y=pd.DataFrame(label,columns=["role"])
X1 = pd.DataFrame(data,columns=['sslc','hsc','cgpa','school_type','no_of_miniprojects','no_of_projects','coresub_skill','aptitude_skill','problemsolving_skill','programming_skill','abstractthink_skill','design_skill','ds_coding','technology_used','hackathon_won','college_performence'])

from sklearn.preprocessing import Normalizer
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score



#PCA
#(2)

#Prob apprach :



X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=0.2,random_state=20)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix
X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=0.3,random_state=10)

#X_train=pd.to_numeric(X_train.values.flatten())

#X_train=X_train.reshape((34,25))

from xgboost.sklearn import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
xgb_y_pred  = clf.predict(X_test)

xgb_cm = confusion_matrix(y_test,xgb_y_pred)
xgb_accuracy = accuracy_score(y_test,xgb_y_pred)
print(xgb_accuracy)

import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
dt=dataset.iloc[:49,]
y = dt['ROLE'].values
X = dt.drop('ROLE', axis=1).values
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X,y)
y_pred = knn.predict(X)
#cm fro knn 
#acc using knn



print(y_pred)

#model save 
outfile=open('model1.pkl','wb')
pickle.dump(knn,outfile)
outfile.close()
