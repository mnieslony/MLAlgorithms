import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import tree
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

import pickle #for saving & loading models

#------- Merge .csv files -------

data = pd.read_csv("data/beam_muon_FV_PMTVol_SingleMultiRing_DigitThr10_wPhi_0_4996.csv",header=0)    #first row is header
data['multiplerings'] = data['multiplerings'].astype('str')
data.replace({'multiplerings':{'0':'1-ring',str(1):'multi-ring'}},inplace=True)
X_test = data.iloc[:,[0,1,2,3,4,5,8,9,12,13,14,15,17,19,20,21,33,34]]
Y_test = data.iloc[:,42:43]  # Classification on 'multiplerings' property of events

print("X_test data: ",X_test)
print("Y_test data: ",Y_test)

feature_labels=list(X_test.columns)
print("Length of feature_labels: %i" %(len(feature_labels)))

scaler = preprocessing.StandardScaler()
X_test = pd.DataFrame(scaler.fit_transform(X_test))

model_names=["RandomForest","XGBoost","GradientBoosting","SVM","SGD","MLP"]

print("Evaluating model performance for ring classifiers")
for imodel in model_names:
        print("Evaluating performance of model ",imodel," on data set...")
        loaded_model = pickle.load(open("models/RingClassification/ringcounting_model_"+imodel+".sav", 'rb'))
        score = loaded_model.score(X_test, Y_test)
        print("Score: ",score)

        Y_pred = loaded_model.predict(X_test)
        accuracy = accuracy_score(Y_test,Y_pred) * 100
        print("Accuracy: %1.3f\n" %accuracy)

        report = classification_report(Y_test,Y_pred)
        print("Report: ",report)

