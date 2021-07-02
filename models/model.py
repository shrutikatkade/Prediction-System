# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 19:40:03 2019

@author: Urjita Kerkar
"""

import pandas as pd
import pickle

dataset = pd.read_csv("datasets/Test.csv")

#data = dataset.iloc[:49,:-1].values
#label = dataset.iloc[:49,-1].values

dt=dataset.iloc[:49,]
y = dt['ROLE'].values
X = dt.drop('ROLE', axis=1).values
#numpy.save

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
label = labelencoder.fit_transform(label)


y=pd.DataFrame(label,columns=["role"])
X1 = pd.DataFrame(X,columns=['sslc','hsc','cgpa','school_type','no_of_miniprojects','no_of_projects','coresub_skill','aptitude_skill','problemsolving_skill','programming_skill','abstractthink_skill','design_skill','ds_coding','technology_used','hackathon_won','college_performence'])

from sklearn.preprocessing import Normalizer
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score

X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=0.3,random_state=20)

clf = tree.DecisionTreeClassifier(random_state=0)
clf = clf.fit(X_train, y_train)
ypred_DecisionTreeClassifier  = clf.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,ypred_DecisionTreeClassifier)
row = len(cm)
count = 0
i=0
while(i<row):
    count = count + cm[i,i]
    i = i +1

DTaccuracy = count/len(y_test)


from xgboost.sklearn import XGBClassifier
xgboostmodel = XGBClassifier(random_state=0)
xgboostmodel.fit(X,y)
ypred_XGBoost = xgboostmodel.predict(X)
cmxgboost = confusion_matrix(y,ypred_XGBoost)
print(ypred_XGBoost)
outfile=open('XGBoostClassifier','wb')
pickle.dump(xgboostmodel,outfile)
outfile.close()


row = len(cmxgboost)
count = 0
i=0
while(i<row):
    count = count + cm[i,i]
    i = i +1

XGBoostaccuracy = count/len(y_test)
###########################################################################
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X,y)
ypred=nb.predict(X)
nbprod=nb.predict_proba(X)

nbprod=nbprod*100
probs = nbprod.tolist() 

nb=0.0
mb=0.0    
list1=probs[0]
def find_len(list1): 
    length = len(list1) 
    
    list1.sort() 
    nb=list1[length-1]   
    mb=list1[length-2]
    print(nb)
    print(mb)    
    
outfile=open('GaussianNB','wb')
pickle.dump(nb,outfile)
outfile.close()


cmnb = confusion_matrix(y_test,nby_pred)
from sklearn import metrics
print(metrics.accuracy_score(y_test,nby_pred)*100)