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

#data_e = pd.read_csv("data/beamlike_electron_FV_MRDCluster_trunc.csv",header=None)
data_e = pd.read_csv("data/beam_electron_FV_MRDCluster_0_4999.csv", header = None)
#data_e = pd.read_csv("data/beamlike_electron_FV_MRDCluster_0_277.csv",header = None)
data_e[31] = "electron"
#data_mu = pd.read_csv("data/beamlike_muon_FV_MRDCluster_trunc.csv",header=None)
data_mu = pd.read_csv("data/beam_muon_FV_MRDCluster_lowstat.csv",header = None)
#data_mu = pd.read_csv("data/beam_muon_FV_MRDCluster_0_4997.csv",header=None)
#data_mu = pd.read_csv("data/beamlike_muon_FV_MRDCluster_0_499.csv", header = None)
data_mu[31] = "muon"

data = pd.concat([data_e,data_mu],axis=0, ignore_index=True)    #ignore_index: one continuous index variable instead of separate ones for the 2 datasets

X_test = data.iloc[:,0:31]  # ignore first column which is row Id, no Var+Skew+Kurt (col 5,6,7)
Y_test = data.iloc[:,31:32]  # Classification on the 'Species'

#specify in string which variables are not used
removedVars = "BeamlikeSample_FV_MRDCluster_NoMC"

print("X_test data: ",X_test)
print("Y_test data: ",Y_test)

feature_labels=["pmt_hits","pmt_totalQ","pmt_avgT","pmt_baryAngle","pmt_rmsAngle","pmt_varAngle","pmt_skewAngle","pmt_kurtAngle","pmt_rmsBary","pmt_varBary","pmt_skewBary","pmt_kurtBary","lappd_hits","lappd_avgT","lappd_baryAngle","lappd_rmsAngle","lappd_varAngle","lappd_skewAngle","lappd_kurtAngle","lappd_rmsBary","lappd_varBary","lappd_skewBary","lappd_kurtBary","pmt_fracHighestQ","pmt_fracDownstream","mrd_paddles","mrd_layers","mrd_conslayers","mrd_cluster"]


cols = [29,30]
X_test.drop(X_test.columns[cols],axis=1,inplace=True)
scaler = preprocessing.StandardScaler()
X_test = pd.DataFrame(scaler.fit_transform(X_test))

#filename = 'finalized_model_DecisionTree.sav'
#filename = 'finalized_model_beamlikeMu_DecisionTree.sav'
filename = 'finalized_modelXGBoost.sav'
#filename = 'finalized_model_beamMu_RandomForest.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

Y_pred = loaded_model.predict(X_test)
accuracy = accuracy_score(Y_test,Y_pred) * 100
print("accuracy: %1.3f\n" %accuracy)

