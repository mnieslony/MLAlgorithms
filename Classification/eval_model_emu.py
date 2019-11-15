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

import pickle #for saving models

#------- Merge .csv files -------

use_lappd_info = 0    #should the model use data from the LAPPDs? 1: yes, 0: no

#------- Merge .csv files -------

data_e = pd.read_csv("data/beamlike_electron_FV_PMTVol_DigitThr10_0_276.csv",header = 0)
data_e['particleType'] = "electron"
data_mu = pd.read_csv("data/beamlike_muon_FV_PMTVol_DigitThr10_0_499.csv",header = 0)
data_mu['particleType'] = "muon"
data = pd.concat([data_e,data_mu],axis=0, ignore_index = True)    #ignore_index: one continuous index variable instead of separate ones for the 2 datasets

#balance data to be 50% electron, 50% muon
balanced_data = data.groupby('particleType')
balanced_data = (balanced_data.apply(lambda x: x.sample(balanced_data.size().min()).reset_index(drop=True)))

if not use_lappd_info:
        X_test = balanced_data.iloc[:,[0,1,2,3,4,5,8,9,12,13,14,15,17,19,20,21,33,34,35,36,37,38,39,40]]
else:
        X_test = balanced_data.iloc[:,[0,1,2,3,4,5,8,9,12,13,14,15,17,19,20,21,22,23,25,26,29,30,33,34,35,36,37,38,39,40]]
Y_test = balanced_data.iloc[:,46:47]  # Classification on the particle type

#specify in string which variables are not used
print("X_test data: ",X_test)
print("Y_test data: ",Y_test)

feature_labels = list(X_test.columns)

scaler = preprocessing.StandardScaler()
X_test = pd.DataFrame(scaler.fit_transform(X_test))

model_names=["RandomForest","XGBoost","GradientBoosting","SVM","SGD","MLP"]
print("Evaluating model performance for PID classifiers")
for imodel in model_names:
    print("Evaluating performance of model ",imodel," on data set...")
    loaded_model = pickle.load(open("models/PID/pid_model_"+imodel+".sav",'rb'))
    score = loaded_model.score(X_test, Y_test)
    print("Score: ",score)

    Y_pred = loaded_model.predict(X_test)
    accuracy = accuracy_score(Y_test,Y_pred) * 100
    print("Accuracy: %1.3f\n" %accuracy)

    report = classification_report(Y_test,Y_pred)
    print("Report: ",report)

