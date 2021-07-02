# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 13:04:32 2019

@author: MY PC
"""

import pandas as pd
import pickle

dataset = pd.read_csv("datasets/Test.csv")

data = dataset.iloc[:49,:-1].values
label = dataset.iloc[:49,-1].values

#numpy.save

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
label = labelencoder.fit_transform(label)

#import numpy
#numpy.save('output', labelencoder.classes_)

y=pd.DataFrame(label,columns=["role"])
X1 = pd.DataFrame(data,columns=['sslc','hsc','cgpa','school_type','no_of_miniprojects','no_of_projects','coresub_skill','aptitude_skill','problemsolving_skill','programming_skill','abstractthink_skill','design_skill','ds_coding','technology_used','hackathon_won','college_performence'])

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)  
X1 = pca.fit_transform(X1)
dbfile=open('preprocess','wb')
pickle.dump(pca.explained_variance_,dbfile)
dbfile.close()

#dividing in training set and test set
X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=0.2,random_state=20)

clf = tree.DecisionTreeClassifier(random_state=0)
clf = clf.fit(X_train, y_train)
ypred_DecisionTreeClassifier  = clf.predict(X_test)

#saving DTclassifier
outfile=open('DecissionTree','wb')
pickle.dump(clf,outfile)
outfile.close()


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
xgboostmodel.fit(X_train, y_train)
ypred_XGBoost = xgboostmodel.predict(X_test)
cmxgboost = confusion_matrix(y_test,ypred_XGBoost)

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


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)
y_predKNN = knn.predict(X_test)

outfile=open('KNNClassifier','wb')
pickle.dump(knn,outfile)
outfile.close()

#cm fro knn 
cmknn = confusion_matrix(y_test,y_predKNN)
row = len(cmknn)
count = 0
i=0
while(i<row):
    count = count + cm[i,i]
    i = i +1
#acc using knn
Knnaccuracy = count/len(y_test)




#nb 
from sklearn.naive_bayes import GaussianNB
nbclassifier = GaussianNB()
nbclassifier.fit(X_train, y_train)
# Predicting the Test set results
nby_pred = nbclassifier.predict(X_test)
nbprod=nbclassifier.predict_proba(X_test)

nbprod=nbprod*100

outfile=open('GaussianNB','wb')
pickle.dump(nbclassifier,outfile)
outfile.close()


cmnb = confusion_matrix(y_test,nby_pred)
from sklearn import metrics
print(metrics.accuracy_score(y_test,nby_pred)*100)
row = len(cmnb)
count = 0
i=0
while(i<row):
    count = count + cm[i,i]
    i = i +1
#acc using knn
nbnaccuracy = count/len(y_test)



print("Accuracy using XGBClassifier Classifier : "+str( XGBoostaccuracy *100 ))
print("Accuracy using KNN Classifier : "+str( Knnaccuracy *100 ))
print("Accuracy using Decision Tree Classifier : "+str( DTaccuracy *100 ))
print("Accuracy using GaussianNB Classifier : "+str( nbnaccuracy *100 ))






#new data:
