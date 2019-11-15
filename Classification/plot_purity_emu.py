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
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier

import pickle #for saving models

use_lappd_info = 0

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
        X = balanced_data.iloc[:,[0,1,2,3,4,5,8,9,12,13,14,15,17,19,20,21,33,34,35,36,37,38,39,40,43]]
else:
        X = balanced_data.iloc[:,[0,1,2,3,4,5,8,9,12,13,14,15,17,19,20,21,22,23,25,26,29,30,33,34,35,36,37,38,39,40,43]]
y = balanced_data.iloc[:,46:47]  # Classification on the particle type

print("X_data: ",X)
print("Y_data: ",y)

feature_labels = list(X.columns)

print("Length of feature_labels: %i" %(len(feature_labels)))
num_plots = int(len(feature_labels)/2)
if (len(feature_labels)%2 == 0):
	num_plots =num_plots - 1

# build train and test dataset
from sklearn.model_selection import train_test_split
X_train0, X_test0, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 100)

# drop the energy variable before starting the training & classification process
cols = 43
X_train0.drop(columns=["energy"],axis=1,inplace=True)
X_test0.drop(columns=["energy"],axis=1,inplace=True)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train0))
X_test = pd.DataFrame(scaler.transform(X_test0))
print("type(X_train) ",type(X_train)," type(X_train0): ",type(X_train0))
y_train2 = np.array(y_train).ravel() #Return a contiguous flattened 1-d array
print("X_train.shape: ",X_train.shape," y_train2.shape: ",y_train2.shape, "X_test.shape: ",X_test.shape," y_test.shape: ",y_test.shape)


def calculate_accuracy(lower_bin,max_bins,out):
        return sum(out[lower_bin:max_bins])/(sum(out[0:max_bins]))

def calculate_accepted(lower_bin,max_bins,out):
        return sum(out[lower_bin:max_bins])

def run_model(model, model_name):
    
    model.fit(X_train, y_train2)
    y_pred = model.predict(X_test)        
    report = classification_report(y_test, y_pred) 
    print(report)
    
def eval_accuracy(model, model_name):

    y_pred = model.predict(X_test)
    accuracy =  accuracy_score(y_test, y_pred) * 100

    y_pred_prob = model.predict_proba(X_test)
    #y_pred_logprob = model.predict_log_proba(X_test)
            
    proba_muon = []
    proba_electron = []
    proba_electron_muon = []
    proba_electron_electron = []
    i=0
    for prob in y_pred_prob:
            if y_test.iloc[i][0]=="muon":
                    proba_muon.append(prob[1])
                    proba_electron_muon.append(prob[0])
            elif y_test.iloc[i][0]=="electron":
                    proba_electron.append(prob[1])
                    proba_electron_electron.append(prob[0])
            i=i+1

    out_electron = plt.hist(proba_electron,bins=101,range=(0,1),label='true = electron')
    out_muon = plt.hist(proba_muon,bins=101,range=(0,1),label='true = muon')
    plt.xlabel("pred probability")
    plt.title(model_name+" Prediction = Muon")
    plt.legend()
    plt.savefig(model_name+"_predictionMuon.pdf",format="pdf")
    plt.clf()

    out_electron_muon = plt.hist(proba_electron_muon,bins=101,range=(0,1),label='true = muon')
    out_electron_electron = plt.hist(proba_electron_electron,bins=101,range=(0,1),label='true = electron')
    plt.xlabel("pred probability")
    plt.title(model_name+" Prediction = Electron")
    plt.legend()
    plt.savefig(model_name+"_predictionElectron.pdf",format="pdf")
    plt.clf()
    
    print("Accuracy muon (50%): ",calculate_accuracy(50,101,out_muon[0]))
    print("Purity muon (50%): ",calculate_accepted(50,101,out_muon[0])/(calculate_accepted(50,101,out_muon[0])+calculate_accepted(50,101,out_electron[0])))
    print("Accuracy electron (50%): ",calculate_accuracy(50,101,out_electron_electron[0]))
    print("Purity electron (50%): ",calculate_accepted(50,101,out_electron_electron[0])/(calculate_accepted(50,101,out_electron_electron[0])+calculate_accepted(50,101,out_electron_muon[0])))   

            
    accuracy_muon = []
    purity_muon = []
    goodness_muon = []
    for bin in range(101):
            accuracy_muon.append(calculate_accuracy(bin,101,out_muon[0]))
            if abs(calculate_accepted(bin,101,out_muon[0]))<0.1 and abs(calculate_accepted(bin,101,out_electron[0]))<0.1:
                purity_muon.append(0)
            else:
                if weighted:
                    purity_muon.append((1-frac_electron)*calculate_accepted(bin,101,out_muon[0])/((1-frac_electron)*calculate_accepted(bin,101,out_muon[0])+frac_electron*calculate_accepted(bin,101,out_electron[0])))
                else:
                    purity_muon.append(calculate_accepted(bin,101,out_muon[0])/(calculate_accepted(bin,101,out_muon[0])+calculate_accepted(bin,101,out_electron[0])))
            goodness_muon.append(accuracy_muon[bin]*purity_muon[bin]*purity_muon[bin])

    x_values = np.arange(0,1.01,0.01)
    plt.plot(x_values,goodness_muon,label='$\mathregular{purity^2}$*efficiency',linestyle='-')
    plt.plot(x_values,accuracy_muon,color='red',label='accuracy',linestyle='-')
    plt.plot(x_values,purity_muon,color='black',label='purity',linestyle='-')
    plt.title(model_name+" PID")
    plt.xlabel("pred probability")
    plt.legend()
    plt.savefig(model_name+"_PID_accuracypurity.pdf",format="pdf")
    plt.clf()


def plot_accuracy(model,model_name,icol,accuracy_model,purity_model,goodness_model,weighted):

    y_pred = model.predict(X_test)
    accuracy =  accuracy_score(y_test, y_pred) * 100 

    y_pred_prob = model.predict_proba(X_test)
    #y_pred_logprob = model.predict_log_proba(X_test)

    proba_muon = []
    proba_electron = []
    proba_electron_muon = []
    proba_electron_electron = []
    i=0
    for prob in y_pred_prob:
            if y_test.iloc[i][0]=="muon":
                    proba_muon.append(prob[1])
                    proba_electron_muon.append(prob[0])
            elif y_test.iloc[i][0]=="electron":
                    proba_electron.append(prob[1])
                    proba_electron_electron.append(prob[0])
            i=i+1
    out_electron = plt.hist(proba_electron,bins=101,range=(0,1),label='true = electron')
    out_muon = plt.hist(proba_muon,bins=101,range=(0,1),label='true = muon')
    plt.clf()
            
    accuracy_muon = []
    purity_muon = []
    goodness_muon = []
    for bin in range(101):
            accuracy_muon.append(calculate_accuracy(bin,101,out_muon[0]))
            if (abs(calculate_accepted(bin,101,out_muon[0]))<0.1) and (abs(calculate_accepted(bin,101,out_electron[0]))<0.1):
                purity_muon.append(0)
            else:
                if weighted:
                    purity_muon.append((1-frac_electron)*calculate_accepted(bin,101,out_muon[0])/((1-frac_electron)*calculate_accepted(bin,101,out_muon[0])+frac_electron*calculate_accepted(bin,101,out_electron[0])))
                else:
                    purity_muon.append(calculate_accepted(bin,101,out_muon[0])/(calculate_accepted(bin,101,out_muon[0])+calculate_accepted(bin,101,out_electron[0])))
            goodness_muon.append(accuracy_muon[bin]*purity_muon[bin]*purity_muon[bin])

    for i in range(len(accuracy_muon)):
            accuracy_model.append(accuracy_muon[i])
            purity_model.append(purity_muon[i])
            goodness_model.append(goodness_muon[i])

    

# ----- Evaluate models, make plots ------------

model_names=["RandomForest","MLP","XGB","SVM","SGD","GradientBoosting"]
model_colors=['blue','black','red','green','purple','grey']
accuracy_models=[]
purity_models=[]
goodness_models=[]
weighted = 0            #calculate accuracy / purity for biased (1) / unbiased (0) data sample (more 1-ring events than multi-ring events)
frac_electron = 0.01       #only to be used when using weighted = 1

#Plot pred comparison accuracy/purity curves between all classifiers

model = RandomForestClassifier(n_estimators=100)
for imodel in range(len(model_names)):
        accuracy_model=[]
        purity_model=[]
        goodness_model=[]
        if model_names[imodel] == "RandomForest":
                model = RandomForestClassifier(n_estimators=100)
        elif model_names[imodel] == "MLP":
                model = MLPClassifier(hidden_layer_sizes= 100, activation='relu')
        elif model_names[imodel] == "XGB":
                model = XGBClassifier(subsample=0.6, n_estimators=100, min_child_weight=5, max_depth=4, learning_rate=0.15, gamma=0.5, colsample_bytree=1.0)
        elif model_names[imodel] == "SVM":
                model = SVC(probability=True)
        elif model_names[imodel] == "SGD":
                model = OneVsRestClassifier(SGDClassifier(loss="log", max_iter=1000))
        elif model_names[imodel] == "GradientBoosting":
                model = GradientBoostingClassifier(learning_rate=0.01, max_depth=5, n_estimators=200)
        run_model(model,model_names[imodel])
        plot_accuracy(model,model_names[imodel],model_colors[imodel],accuracy_model,purity_model,goodness_model,weighted)
        accuracy_models.append(accuracy_model)
        purity_models.append(purity_model)
        goodness_models.append(goodness_model)
        eval_accuracy(model,model_names[imodel])


x_values = np.arange(0,1.01,0.01)
for imodel in range(len(model_names)):
        plt.plot(x_values,accuracy_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("accuracy")
plt.title("Accuracy comparison - PID")
plt.legend()
if weighted:
        plt.savefig("PID_AccuracyComparison_RealisticComposition.pdf",format="pdf")
else:
        plt.savefig("PID_AccuracyComparison.pdf",format="pdf")
plt.clf()

for imodel in range(len(model_names)):
        plt.plot(x_values,purity_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("purity")
plt.title("Purity comparison - PID")
plt.legend()
if weighted:
        plt.savefig("PID_PurityComparison_RealisticComposition.pdf",format="pdf")
else:
        plt.savefig("PID_PurityComparison.pdf",format="pdf")
plt.clf()

for imodel in range(len(model_names)):
        plt.plot(x_values,goodness_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("$accuracy*purity^2$")
plt.title("$Accuracy*Purity^2$ comparison - PID")
plt.legend()
if weighted:
        plt.savefig("PID_AccuracyPurity2Comparison_RealisticComposition.pdf",format="pdf")
else:
        plt.savefig("PID_AccuracyPurity2Comparison.pdf",format="pdf")
plt.clf()



