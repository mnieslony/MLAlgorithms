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


#------- Merge .csv files -------

data = pd.read_csv("data/beam_muon_FV_PMTVol_SingleMultiRing_DigitThr10_wPhi_0_4996.csv",header = 0)   #first row is header

data['multiplerings'] = data['multiplerings'].astype('str')
data.replace({'multiplerings':{'0':'1-ring',str(1):'multi-ring'}},inplace=True)

balanced_data = data.groupby('multiplerings')
balanced_data = (balanced_data.apply(lambda x: x.sample(balanced_data.size().min()).reset_index(drop=True)))

X = balanced_data.iloc[:,[0,1,2,3,4,5,8,9,12,13,14,15,17,19,20,21,33,34,43]]   # ignore first column which is row Id
y = balanced_data.iloc[:,42:43]  # Classification on the boolean 'multiplerings'

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


def calculate_accuracy(bin_width,lower_bin,max_bins,out):
        return sum(out[lower_bin:max_bins])/(sum(out[0:max_bins]))

def calculate_accepted(bin_width,lower_bin,max_bins,out):
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

    proba_singlering = []
    proba_multiring = []
    proba_multi_singlering = []
    proba_multi_multiring = []
    i=0
    for prob in y_pred_prob:
            if y_test.iloc[i][0]=="1-ring":
                    proba_singlering.append(prob[0])
                    proba_multi_singlering.append(prob[1])
            elif y_test.iloc[i][0]=="multi-ring":
                    proba_multiring.append(prob[0])
                    proba_multi_multiring.append(prob[1])
            i=i+1    
    out_single = plt.hist(proba_singlering,bins=100,label='true = single ring')
    out_multi = plt.hist(proba_multiring,bins=100,label='true = multi ring')
    plt.xlabel("pred probability")
    plt.title(model_name+" Prediction = SingleRing")
    plt.legend()
    plt.savefig(model_name+"_predictionSingleRing.pdf",format="pdf")
    plt.clf()

    out_multi_single = plt.hist(proba_multi_singlering,bins=100,label='true = single ring')
    out_multi_multi = plt.hist(proba_multi_multiring,bins=100,label='true = multi ring')
    plt.xlabel("pred probability")
    plt.title("RandomForest Prediction = MultiRing")
    plt.legend()
    plt.savefig(model_name+"_predictionMultiRing.pdf",format="pdf")
    plt.clf()
    

    bin_width = out_single[1][1]-out_single[1][0]
    print("Integral single ring: 0.5 - 1.0 (w function): ",calculate_accuracy(bin_width,50,99,out_single[0]))
    print("Accepted single rings (50%): ",calculate_accepted(bin_width,50,99,out_single[0]))
    print("Accepted multi rings (50%): ",calculate_accepted(bin_width,50,99,out_multi[0]))
    print("Purity (50%): ",calculate_accepted(bin_width,50,99,out_single[0])/(calculate_accepted(bin_width,50,99,out_single[0])+calculate_accepted(bin_width,50,99,out_multi[0])))

            
    accuracy_single = []
    purity_single = []
    goodness_single = []
    for bin in range(100):
            accuracy_single.append(calculate_accuracy(bin_width,bin,99,out_single[0]))
            if bin!=99:
                    purity_single.append(calculate_accepted(bin_width,bin,99,out_single[0])/(calculate_accepted(bin_width,bin,99,out_single[0])+calculate_accepted(bin_width,bin,99,out_multi[0])))
            else:
                    purity_single.append(0.)
            goodness_single.append(accuracy_single[bin]*purity_single[bin]*purity_single[bin])

    x_values = np.arange(0,1,0.01)
    plt.plot(x_values,goodness_single,label='$\mathregular{purity^2}$*efficiency',linestyle='-')
    plt.plot(x_values,accuracy_single,color='red',label='accuracy',linestyle='-')
    plt.plot(x_values,purity_single,color='black',label='purity',linestyle='-')
    plt.title(model_name+" Ring Classification")
    plt.xlabel("pred probability")
    plt.legend()
    plt.savefig(model_name+"_SingleRing_accuracypurity.pdf",format="pdf")
    plt.clf()


def plot_accuracy(model,model_name,icol,accuracy_model,purity_model,goodness_model,weighted):

    y_pred = model.predict(X_test)
    accuracy =  accuracy_score(y_test, y_pred) * 100 

    y_pred_prob = model.predict_proba(X_test)
    #y_pred_logprob = model.predict_log_proba(X_test)

    proba_singlering = []
    proba_multiring = []
    proba_multi_singlering = []
    proba_multi_multiring = []
    i=0
    for prob in y_pred_prob:
            if y_test.iloc[i][0]=="1-ring":
                    proba_singlering.append(prob[0])
                    proba_multi_singlering.append(prob[1])
            elif y_test.iloc[i][0]=="multi-ring":
                    proba_multiring.append(prob[0])
                    proba_multi_multiring.append(prob[1])
            i=i+1    
    out_single = plt.hist(proba_singlering,bins=100,label='true = single ring')
    out_multi = plt.hist(proba_multiring,bins=100,label='true = multi ring')
    plt.clf()

    bin_width = out_single[1][1]-out_single[1][0]
    accuracy_single = []
    purity_single = []
    goodness_single = []
    for bin in range(100):
            accuracy_single.append(calculate_accuracy(bin_width,bin,99,out_single[0]))
            if bin!=99:
                    if weighted:
                            purity_single.append((1-frac_multi)*calculate_accepted(bin_width,bin,99,out_single[0])/((1-frac_multi)*calculate_accepted(bin_width,bin,99,out_single[0])+frac_multi*calculate_accepted(bin_width,bin,99,out_multi[0])))
                    else:
                            purity_single.append(calculate_accepted(bin_width,bin,99,out_single[0])/(calculate_accepted(bin_width,bin,99,out_single[0])+calculate_accepted(bin_width,bin,99,out_multi[0])))
            else:
                    purity_single.append(0.)
            goodness_single.append(accuracy_single[bin]*purity_single[bin]*purity_single[bin])

    for i in range(len(accuracy_single)):
            accuracy_model.append(accuracy_single[i])
            purity_model.append(purity_single[i])
            goodness_model.append(goodness_single[i])

    

# ----- Evaluate models, make plots ------------

model_names=["RandomForest","MLP","XGB","SVM","SGD","GradientBoosting"]
model_colors=['blue','black','red','green','purple','grey']
accuracy_models=[]
purity_models=[]
goodness_models=[]
weighted = 0            #calculate accuracy / purity for biased (1) / unbiased (0) data sample (more 1-ring events than multi-ring events)
frac_multi = 0.37       #only to be used when using weighted = 1

#Plot pred comparison accuracy/purity curves between all classifiers

model = RandomForestClassifier(n_estimators=100)
for imodel in range(len(model_names)):
        accuracy_model=[]
        purity_model=[]
        goodness_model=[]
        #pick_model(model,model_names[imodel])
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

x_values = np.arange(0,1,0.01)
for imodel in range(len(model_names)):
        plt.plot(x_values,accuracy_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("accuracy")
plt.title("Accuracy comparison - Ring Classification")
plt.legend()
if weighted:
        plt.savefig("RingClassification_AccuracyComparison_RealisticComposition.pdf",format="pdf")
else:
        plt.savefig("RingClassification_AccuracyComparison.pdf",format="pdf")
plt.clf()

for imodel in range(len(model_names)):
        plt.plot(x_values,purity_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("purity")
plt.title("Purity comparison - Ring Classification")
plt.legend()
if weighted:
        plt.savefig("RingClassification_PurityComparison_RealisticComposition.pdf",format="pdf")
else:
        plt.savefig("RingClassification_PurityComparison.pdf",format="pdf")
plt.clf()

for imodel in range(len(model_names)):
        plt.plot(x_values,goodness_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("$accuracy*purity^2$")
plt.title("$Accuracy*Purity^2$ comparison - Ring Classification")
plt.legend()
if weighted:
        plt.savefig("RingClassification_AccuracyPurity2Comparison_RealisticComposition.pdf",format="pdf")
else:
        plt.savefig("RingClassification_AccuracyPurity2Comparison.pdf",format="pdf")
plt.clf()

#Plot pred probability histograms & associated purity and accuracy curves

for imodel in range(len(model_names)):
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
        eval_accuracy(model,model_names[imodel])



