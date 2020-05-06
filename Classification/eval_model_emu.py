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

import pickle #for loading/saving models

import argparse #For user input

#------- Parse user arguments ----

parser = argparse.ArgumentParser(description='PID Model Evaluation - Overview')
parser.add_argument("--input_e", default="data.nosync/beamlike_electron_DigitThr10_0_276_Full.csv", help = "The input electron file containing the data to be evaluated [csv-format]")
parser.add_argument("--input_mu", default="data.nosync/beamlike_muon_Digitthr10_0_498_Full.csv", help = "The input muon file containing the data to be evaluated [csv-format]")
parser.add_argument("--variable_names", default="VariableConfig_Full.txt", help = "File containing the list of classification variables")
parser.add_argument("--model",default="models/PID/pid_model_MLP_beamlike_Full.sav",help="Path to classification model")
parser.add_argument("--balance_data",default=False,help="Should the evaluated input files have balanced classes?")
args = parser.parse_args()
input_file_e = args.input_e
input_file_mu = args.input_mu
variable_file = args.variable_names
model_file = args.model
balance_data = args.balance_data

print('PID classification evaluation: Input_file (electron): '+input_file_e+', input file (muon): '+input_file_mu+', variable file: '+variable_file+', model_file: '+model_file)


#------- Merge .csv files -------

data_e = pd.read_csv(input_file_e,header = 0)
data_e['particleType'] = "electron"
data_mu = pd.read_csv(input_file_mu,header = 0)
data_mu['particleType'] = "muon"
data = pd.concat([data_e,data_mu],axis=0, ignore_index = True)    #ignore_index: one continuous index variable instead of separate ones for the 2 datasets

if balance_data:
    #balance data to be 50% electron, 50% muon
    balanced_data = data.groupby('particleType')
    balanced_data = (balanced_data.apply(lambda x: x.sample(balanced_data.size().min()).reset_index(drop=True)))
else:
    balanced_data = data

#---------- Load only specific variables of data ---------

with open(variable_file) as f:
    subset_variables = f.read().splitlines()
subset_variables.append('particleType')

balanced_data = balanced_data[subset_variables]


# ----- Load model & variables associated to model -------

loaded_model, feature_vars, loaded_scaler = pickle.load(open(model_file,'rb'))

# ------------- Number of variables check --------------

eval_model = True
if len(feature_vars) != len(subset_variables):
    print('Number of variables does not match in model & data (Model: '+str(len(feature_vars))+', Data: '+str(len(subset_variables))+'). Abort')
    eval_model = False

if eval_model:
    for i in range(len(feature_vars)):

        if feature_vars[i] != subset_variables[i]:
            print('Variables at index '+str(i)+' do not match in model & data (Model: '+feature_vars[i]+', Data: '+subset_variables[i]+'). Abort')
            eval_model = False


#-------Evaluate model performance on data set------

if eval_model:
    X_test = balanced_data.loc[:,balanced_data.columns!='particleType']
    Y_test = balanced_data.iloc[:,-1]

    X_test = pd.DataFrame(loaded_scaler.transform(X_test))

    print("Evaluating model performance for PID e/mu classifier ",model_file," on data set...")

    score = loaded_model.score(X_test,Y_test)
    print("Score: ",score)

    Y_pred = loaded_model.predict(X_test)
    accuracy = accuracy_score(Y_test,Y_pred) * 100
    print("Accuracy: %1.3f\n" %accuracy) 

    report = classification_report(Y_test,Y_pred)
    print("Report: ",report)


