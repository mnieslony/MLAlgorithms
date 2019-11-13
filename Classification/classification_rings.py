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
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
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

#read in data file
data = pd.read_csv("data/beam_muon_FV_PMTVol_SingleMultiRing_DigitThr10_wPhi_0_4996.csv",header = 0)   #first row is header

data['multiplerings'] = data['multiplerings'].astype('str')
data.replace({'multiplerings':{'0':'1-ring',str(1):'multi-ring'}},inplace=True)

#balance data to be 50% single ring, 50% multi ring
balanced_data = data.groupby('multiplerings')
balanced_data = (balanced_data.apply(lambda x: x.sample(balanced_data.size().min()).reset_index(drop=True)))

X = balanced_data.iloc[:,[0,1,2,3,4,5,8,9,12,13,14,15,17,19,20,21,33,34,43]]   # ignore first column which is row Id
y = balanced_data.iloc[:,42:43]  # Classification on the boolean 'multiplerings'

#specify in string which variables are not used
removedVars = "Beam_FV_PMTVol_DigitThr10"

print("X_data: ",X)
print("Y_data: ",y)

#get feature labels from header
feature_labels = list(X.columns)

print("Length of feature_labels: %i" %(len(feature_labels)))
num_plots = int(len(feature_labels)/2)
if (len(feature_labels)%2 == 0):
	num_plots =num_plots - 1

# build train and test dataset
from sklearn.model_selection import train_test_split
X_train0, X_test0, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 100)

#retrieving information about which events are in test dataset --> original indices
X_test_indices = X_test0.index.get_level_values(1)
print(X_test_indices)
file = open("RingCounting_Indices_"+removedVars+".dat","w")
for index in X_test_indices:
    file.write("%i\n" % index)
file.close()

# save information about events
X_test0.to_csv('RingCounting_AddEvInfo_'+removedVars+'.csv')
print("X_train0: ",X_train0)

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


# Setup additional data sets with labels 0/1 instead of 1-ring/multi-ring
y_test_binary = y_test
np.where(y_test_binary == '1-ring', 0, 1)

fig=[]
for plot in range(num_plots):
	fig.append(plt.figure(plot,figsize=(15,20)))
fig_roc = plt.figure(num_plots+1,figsize=(15,20))
fig_pr = plt.figure(num_plots+2,figsize=(15,20))

def run_model(model, alg_name):
    
    model.fit(X_train, y_train2)
    y_pred = model.predict(X_test)
    y_pred_binary = y_pred
    np.where(y_pred_binary == '1-ring', 0, 1)
    accuracy =  accuracy_score(y_test, y_pred) * 100 #returns the fraction of correctly classified samples
    print("--------- alg_name: ",alg_name)
    file = open("Accuracy_"+removedVars+".dat","a")
    file.write("%s \t %1.3f \n" % (alg_name, accuracy))
    file.close()

    #write predictions to a file
    predictions = pd.concat([y_test.reset_index(drop=True) ,pd.DataFrame(y_pred,columns=['Prediction'])], axis=1)
    if alg_name=="XGBoost":
       predictions.to_csv('RingCounting_XGBoost_predictions_'+removedVars+'.csv')
    elif alg_name=="Random Forest":
        predictions.to_csv('RingCounting_RandomForest_predictions_'+removedVars+'.csv')
    elif alg_name=="SVM Classifier":
        predictions.to_csv('RingCounting_SVM_predictions_'+removedVars+'.csv')
    elif alg_name=="MLP Neural network":
        predictions.to_csv('RingCounting_MLP_predictions_'+removedVars+'.csv')
    elif alg_name=="GradientBoostingClassifier":
        predictions.to_csv('RingCounting_GradientBoosting_predictions_'+removedVars+'.csv')
    elif alg_name=="SGD Classifier":
        predictions.to_csv('RingCounting_SGD_predictions_'+removedVars+'.csv')
        
    report = classification_report(y_test, y_pred) 
    print(report)

def plot_accuracy(model, alg_name, plot_index, it):
    model.fit(X_train, y_train2)
    y_pred = model.predict(X_test)
    accuracy =  accuracy_score(y_test, y_pred) * 100 
    predictions = pd.concat([y_test.reset_index(drop=True) ,pd.DataFrame(y_pred,columns=['Prediction'])], axis=1)

    #make scatter plot of predictions as a function of 2 feature variables
    color_code = {'1-ring':'red', 'multi-ring':'blue'}
    ax = fig[it].add_subplot(3,2,plot_index) #nrows, ncols, index
    colors = [color_code[x] for x in y_test.iloc[:,0]]
    ax.scatter(X_test.iloc[:,2*it], X_test.iloc[:,2*it+1], color=colors, marker='.', label='Circle = Ground truth')
    colors = [color_code[x] for x in y_pred]
    ax.scatter(X_test.iloc[:,2*it], X_test.iloc[:,2*it+1], color=colors, marker='o', facecolors='none', label='Dot = Prediction')
    ax.legend(loc="lower right")
    leg = plt.gca().get_legend()
    ax.set_title(alg_name + ". Accuracy: " + str(accuracy))
    ax.set_xlabel(feature_labels[2*it])
    ax.set_ylabel(feature_labels[2*it+1])
    if alg_name=="XGBoost":
           filename = 'ringcounting_model_XGB.sav'
    elif alg_name=="Random Forest":
            filename = 'ringcounting_model_RandomForest.sav'
    elif alg_name == "SVM Classifier":
            filename = 'ringcounting_model_SVM.sav'
    elif alg_name == "MLP Neural network":
            filename = 'ringcounting_model_MLP.sav'
    elif alg_name == "GradientBoostingClassifier":
            filename = 'ringcounting_model_GradientBoosting.sav'
    elif alg_name == "SGD Classifier":
            filename = 'ringcounting_model_SGD.sav'
    # save the model to disk
    pickle.dump(model, open(filename, 'wb'))


def plot_roc(model, alg_name, plot_index):
    #Calculate and Plot ROC curve
   
    model.probability = True
    probas = model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 0], pos_label ="1-ring") #assumes single ring
    roc_auc  = auc(fpr, tpr)
 
    ax_roc = fig_roc.add_subplot(3,2,plot_index)
    ax_roc.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (alg_name, roc_auc))
    ax_roc.plot([0, 1], [0, 1], 'k--')          #no skill as comparison
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc=0, fontsize='small')

def plot_precision_recall(model, alg_name, plot_index):
    #Calculate and Plot Precision-Recall curve

    model.probability = True
    probas = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    precision, recall, _ = precision_recall_curve(y_test, probas[:,0],pos_label = "1-ring")
    pr_auc = auc(recall,precision)
    pr_f1 = f1_score(y_test,y_pred,pos_label="1-ring")
    
    ax_pr = fig_pr.add_subplot(3,2,plot_index)
    ax_pr.plot(recall,precision,label='AUC = %0.2f, f1_score = %0.2f' % (pr_auc,pr_f1))
    ax_pr.set_title('%s Ring Classification' % (alg_name))
    ax_pr.plot([0,1],[1,0.5],'k--')     #no skill as comparison
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.legend()


#------ Test different models:

# ----- Random Forest ---------------

model = RandomForestClassifier(n_estimators=100)
run_model(model, "Random Forest")
for plot in range(num_plots):
	plot_accuracy(model,"Random Forest",1,plot)
plot_roc(model, "Random Forest",1)
plot_precision_recall(model, "Random Forest",1)

from xgboost import XGBClassifier

model = XGBClassifier(subsample=0.6, n_estimators=100, min_child_weight=5, max_depth=4, learning_rate=0.15, gamma=0.5, colsample_bytree=1.0)
run_model(model, "XGBoost")
for plot in range(num_plots):
	plot_accuracy(model, "XGBoost",2,plot)
plot_roc(model, "XGBoost",2)
plot_precision_recall(model, "XGBoost",2)

# ------ SVM Classifier ----------------
model = SVC(probability=True)
run_model(model, "SVM Classifier")
for plot in range(num_plots):
	plot_accuracy(model, "SVM Classifier",3,plot)
plot_roc(model, "SVM Classifier",3)
plot_precision_recall(model, "SVM Classifier",3)

# ---------- SGD Classifier -----------------

model = OneVsRestClassifier(SGDClassifier(loss="log", max_iter=1000))
run_model(model, "SGD Classifier")
for plot in range(num_plots):
	plot_accuracy(model, "SGD Classifier",4,plot)
plot_roc(model, "SGD Classifier",4)
plot_precision_recall(model, "SGD Classifier",4)

# ----------- Neural network - Multi-layer Perceptron  ------------

model = MLPClassifier()
model = MLPClassifier(hidden_layer_sizes= 100, activation='relu')
run_model(model, "MLP Neural network")
for plot in range(num_plots):
	plot_accuracy(model,"MLP Neural network",5,plot)
plot_roc(model, "MLP Neural network",5)
plot_precision_recall(model, "MLP Neural network",5)


#----------- GradientBoostingClassifier -----------

model = GradientBoostingClassifier(learning_rate=0.01, max_depth=5, n_estimators=200)
run_model(model, "GradientBoostingClassifier")
for plot in range(num_plots):
	plot_accuracy(model,"GradientBoostingClassifier",6,plot)
plot_roc(model, "GradienBoostingClassifier",6)
plot_precision_recall(model, "GradientBoostingClassifier",6)


for plot in range(num_plots):
	fig[plot].savefig("RingCounting_"+removedVars+"_"+feature_labels[2*plot]+"_"+feature_labels[2*plot+1]+".png")

fig_roc.savefig("RingCounting_ROCcurve_"+removedVars+".png")
fig_pr.savefig("RingCounting_PrecRecallcurve_"+removedVars+".png")


######################################################
########## NOT SO WELL PERFORMING MODELS--> BACKUP####
######################################################

# ---- Decision Tree -----------

#model = tree.DecisionTreeClassifier(max_depth=10)
#run_model(model, "Decision Tree")
#for plot in range(num_plots):
#	plot_accuracy(model,"Decision Tree",7,plot)
#plot_roc(model, "Decision Tree",7)

#----------- LogisticRegression ------------

#model = LogisticRegression(penalty='l1', tol=0.01) 
#run_model(model, "LogisticRegression")
#for plot in range(num_plots):
#	plot_accuracy(model,"Logistic Regression",8,plot)
#plot_roc(model, "LogisticRegression",8)

# -------- Nearest Neighbors ----------
#model = neighbors.KNeighborsClassifier(n_neighbors=10)
#run_model(model, "Nearest Neighbors Classifier")
#for plot in range(num_plots):
#	plot_accuracy(model, "Nearest Neighbors Classifier",9,plot)
#plot_roc(model, "Nearest Neighbors Classifier",9)

# --------- Gaussian Naive Bayes ---------

#model = GaussianNB()
#run_model(model, "Gaussian Naive Bayes")
#for plot in range(num_plots):
#	plot_accuracy(model, "Gaussian Naive Bayes",10,plot)
#plot_roc(model, "Gaussian Naive Bayes",10)

