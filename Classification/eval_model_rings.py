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

import argparse #For user input

#------- Parse user arguments ----

parser = argparse.ArgumentParser(description='Ring Classification Model Evaluation - Overview')
parser.add_argument("--input", default="data.nosync/beam_muon_FV_PMTVol_SingleMultiRing_DigitThr10_wPhi_0_4996_Old.csv", help = "The input electron file containing the data to be evaluated [csv-format]")
parser.add_argument("--variable_names", default="VariableConfig_Old.txt", help = "File containing the list of classification variables")
parser.add_argument("--model",default="models/RingClassification/ringcounting_model_beam_Old_MLP.sav",help="Path to classification model")
parser.add_argument("--balance_data",default=False,help="Should the evaluated input files have balanced classes?")
args = parser.parse_args()
input_file = args.input
variable_file = args.variable_names
model_file = args.model
balance_data = args.balance_data

print('Ring Classification evaluation: Input_file: '+input_file+', variable file: '+variable_file+', model_file: '+model_file)


#------- Merge .csv files -------

data = pd.read_csv(input_file,header=0)    #first row is header
data['multiplerings'] = data['multiplerings'].astype('str')
data.replace({'multiplerings':{'0':'1-ring',str(1):'multi-ring'}},inplace=True)

#------- Balance data (if specified) -------

if balance_data:
	#balance data to be 50% single-rings, 50% multi-rings
    balanced_data = data.groupby('multiplerings')
    balanced_data = (balanced_data.apply(lambda x: x.sample(balanced_data.size().min()).reset_index(drop=True)))
else:
    balanced_data = data

 #---------- Load only specific variables of data ---------

with open(variable_file) as f:
    subset_variables = f.read().splitlines()
subset_variables.append('multiplerings')

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

	X_test = balanced_data.loc[:,balanced_data.columns!='multiplerings']
	Y_test = balanced_data.iloc[:,-1]

	X_test = pd.DataFrame(loaded_scaler.transform(X_test))

	print("Evaluating model performance for RingClassification classifier ",model_file," on data set...")

	score = loaded_model.score(X_test, Y_test)
	print("Score: ",score)

	Y_pred = loaded_model.predict(X_test)
	accuracy = accuracy_score(Y_test,Y_pred) * 100
	print("Accuracy: %1.3f\n" %accuracy)

	report = classification_report(Y_test,Y_pred)
	print("Report: ",report)

