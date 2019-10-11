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

data_e = pd.read_csv("data/uniform_electron_FV_1_106_trunc.csv",header = None)
#data_e = pd.read_csv("data/beamlike_electron_FV_MRDCluster_0_268.csv", header = None)
#data_e = pd.read_csv("data/beamlike_electron_FV_MRDCluster_0_277.csv", header = None)
#data_e = pd.read_csv("data/uniform_electron_noFV_125_130.csv", header = None)
#data_e = pd.read_csv("data/electron_nomc.csv", header = None)
#data_e = pd.read_csv("data/beam_electron_noFV_0_1100.csv", header = None)
data_e[31] = "electron"
data_mu = pd.read_csv("data/uniform_muon_FV_1_30.csv",header = None)
#data_mu = pd.read_csv("data/beamlike_muon_FV_MRDCluster_elestat.csv",header=None)
#data_mu = pd.read_csv("data/beamlike_muon_FV_MRDCluster_0_499.csv",header=None)
#data_mu = pd.read_csv("data/beam_muon_FV_MRDCluster_0_4997.csv", header = None)
#data_mu = pd.read_csv("data/uniform_muon_noFV_125_130.csv", header = None)
#data_mu = pd.read_csv("data/muon_nomc.csv", header = None)
#data_mu = pd.read_csv("data/beam_muon_noFV_2_13.csv", header = None)
data_mu[31] = "muon"

data = pd.concat([data_e,data_mu],axis=0, ignore_index=True)    #ignore_index: one continuous index variable instead of separate ones for the 2 datasets

X = data.iloc[:,0:31]  # ignore first column which is row Id, no Var+Skew+Kurt (col 5,6,7)
y = data.iloc[:,31:32]  # Classification on the 'Species'

#specify in string which variables are not used
removedVars = "Uniform_FV_MRDCluster_NoMC"

print("X_data: ",X)
print("Y_data: ",y)

feature_labels=["pmt_hits","pmt_totalQ","pmt_avgT","pmt_baryAngle","pmt_rmsAngle","pmt_varAngle","pmt_skewAngle","pmt_kurtAngle","pmt_rmsBary","pmt_varBary","pmt_skewBary","pmt_kurtBary","lappd_hits","lappd_avgT","lappd_baryAngle","lappd_rmsAngle","lappd_varAngle","lappd_skewAngle","lappd_kurtAngle","lappd_rmsBary","lappd_varBary","lappd_skewBary","lappd_kurtBary","pmt_fracHighestQ","pmt_fracDownstream","mrd_paddles","mrd_layers","mrd_conslayers","mrd_cluster"]
#feature_labels=["pmt_hits","pmt_totalQ","pmt_avgT","pmt_baryAngle","pmt_rmsAngle","pmt_varAngle","pmt_skewAngle","pmt_kurtAngle","pmt_rmsBary","pmt_varBary","pmt_skewBary","pmt_kurtBary","lappd_hits","lappd_avgT","lappd_baryAngle","lappd_rmsAngle","lappd_varAngle","lappd_skewAngle","lappd_kurtAngle","lappd_rmsBary","lappd_varBary","lappd_skewBary","lappd_kurtBary","pmt_fracHighestQ","pmt_fracDownstream"]

print("Length of feature_labels: %i" %(len(feature_labels)))
num_plots = int(len(feature_labels)/2)
if (len(feature_labels)%2 == 0):
	num_plots =num_plots - 1

# build train and test dataset
from sklearn.model_selection import train_test_split
X_train0, X_test0, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 100)

#retrieving information about which events are in test dataset --> original indices

X_test_indices = X_test0.index
print(X_test_indices)
file = open("ShuffleIndices_"+removedVars+".dat","w")
for index in X_test_indices:
    file.write("%i\n" % index)
file.close()

# save information about events
X_test0.to_csv('AddEvInfo_'+removedVars+'.csv')

cols = [29,30]
X_train0.drop(X_train0.columns[cols],axis=1,inplace=True)
X_test0.drop(X_test0.columns[cols],axis=1,inplace=True)

print("X_train0: ",X_train0)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train0))
X_test = pd.DataFrame(scaler.transform(X_test0))

print("type(X_train) ",type(X_train)," type(X_train0): ",type(X_train0))

#X_train2 = np.array(X_train)  # ignore first column which is row Id
y_train2 = np.array(y_train).ravel() #Return a contiguous flattened 1-d array
print("X_train.shape: ",X_train.shape," y_train2.shape: ",y_train2.shape, "X_test.shape: ",X_test.shape," y_test.shape: ",y_test.shape)

fig=[]
for plot in range(num_plots):
	fig.append(plt.figure(plot,figsize=(15,20)))
fig_roc = plt.figure(num_plots+1,figsize=(15,20))

def run_model(model, alg_name):
    
    model.fit(X_train, y_train2)
    y_pred = model.predict(X_test)
    accuracy =  accuracy_score(y_test, y_pred) * 100 #returns the fraction of correctly classified samples
    print("--------- alg_name: ",alg_name)
    file = open("Accuracy_"+removedVars+".dat","a")
    file.write("%s \t %1.3f \n" % (alg_name, accuracy))
    file.close()
    
    predictions = pd.concat([y_test.reset_index(drop=True) ,pd.DataFrame(y_pred,columns=['Prediction'])], axis=1)
    if alg_name=="XGBoost":
       predictions.to_csv('XGBoost_predictions_'+removedVars+'.csv')
    elif alg_name=="Decision Tree":
        predictions.to_csv('DecisionTree_predictions_'+removedVars+'.csv')
    elif alg_name=="Random Forest":
        predictions.to_csv('RandomForest_predictions_'+removedVars+'.csv')
    elif alg_name=="SVM Classifier":
        predictions.to_csv('SVM_predictions_'+removedVars+'.csv')
    elif alg_name=="MLP Neural network":
        predictions.to_csv('MLP_predictions_'+removedVars+'.csv')
    elif alg_name=="GradientBoostingClassifier":
        predictions.to_csv('GradientBoosting_predictions_'+removedVars+'.csv')
        
    report = classification_report(y_test, y_pred) 
    print(report)

def plot_accuracy(model, alg_name, plot_index, it):
    model.fit(X_train, y_train2)
    y_pred = model.predict(X_test)
    accuracy =  accuracy_score(y_test, y_pred) * 100 
    predictions = pd.concat([y_test.reset_index(drop=True) ,pd.DataFrame(y_pred,columns=['Prediction'])], axis=1)
    color_code = {'muon':'red', 'electron':'blue'}
    ax = fig[it].add_subplot(5,2,plot_index) #nrows, ncols, index
    colors = [color_code[x] for x in y_test.iloc[:,0]]
    ax.scatter(X_test.iloc[:,2*it], X_test.iloc[:,2*it+1], color=colors, marker='.', label='Circle = Ground truth')
    colors = [color_code[x] for x in y_pred]
    ax.scatter(X_test.iloc[:,2*it], X_test.iloc[:,2*it+1], color=colors, marker='o', facecolors='none', label='Dot = Prediction')
    ax.legend(loc="lower right")
    leg = plt.gca().get_legend()
    ax.set_title(alg_name + ". Accuracy: " + str(accuracy))
    ax.set_xlabel(feature_labels[2*it])
    ax.set_ylabel(feature_labels[2*it+1])
    # save the model to disk
    filename = 'finalized_model'+alg_name+'.sav'
    pickle.dump(model, open(filename, 'wb'))


def plot_roc(model, alg_name, plot_index):
    #Calculate and Plot ROC curve
   
    model.probability = True
    probas = model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1], pos_label ="muon") #assumes muon is the positive result
    roc_auc  = auc(fpr, tpr)
 
    ax_roc = fig_roc.add_subplot(5,2,plot_index)
    ax_roc.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (alg_name, roc_auc))
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc=0, fontsize='small')

#------ Test different models:
# ---- Decision Tree -----------

model = tree.DecisionTreeClassifier(max_depth=10)
run_model(model, "Decision Tree")
for plot in range(num_plots):
	plot_accuracy(model,"Decision Tree",1,plot)
plot_roc(model, "Decision Tree",1)

# ----- Random Forest ---------------

model = RandomForestClassifier(n_estimators=100)
run_model(model, "Random Forest")
for plot in range(num_plots):
	plot_accuracy(model,"Random Forest",2,plot)
plot_roc(model, "Random Forest",2)

from xgboost import XGBClassifier

#model = XGBClassifier()
#model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#       colsample_bytree=1.0, gamma=2, learning_rate=0.02, max_delta_step=0,
#       max_depth=5, min_child_weight=5, missing=None, n_estimators=600,
#       n_jobs=1, nthread=1, objective='binary:logistic', random_state=0,
#       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#       silent=True, subsample=0.8)
model = XGBClassifier(subsample=0.6, n_estimators=100, min_child_weight=5, max_depth=4, learning_rate=0.15, gamma=0.5, colsample_bytree=1.0)
run_model(model, "XGBoost")
for plot in range(num_plots):
	plot_accuracy(model, "XGBoost",3,plot)
plot_roc(model, "XGBoost",3)

# ------ SVM Classifier ----------------
model = SVC(probability=True)
run_model(model, "SVM Classifier")
for plot in range(num_plots):
	plot_accuracy(model, "SVM Classifier",4,plot)
plot_roc(model, "SVM Classifier",4)

# -------- Nearest Neighbors ----------
model = neighbors.KNeighborsClassifier(n_neighbors=10)
run_model(model, "Nearest Neighbors Classifier")
for plot in range(num_plots):
	plot_accuracy(model, "Nearest Neighbors Classifier",5,plot)
plot_roc(model, "Nearest Neighbors Classifier",5)

# ---------- SGD Classifier -----------------

model = OneVsRestClassifier(SGDClassifier(loss="log", max_iter=1000))
run_model(model, "SGD Classifier")
for plot in range(num_plots):
	plot_accuracy(model, "SGD Classifier",6,plot)
plot_roc(model, "SGD Classifier",6)

# --------- Gaussian Naive Bayes ---------

model = GaussianNB()
run_model(model, "Gaussian Naive Bayes")
for plot in range(num_plots):
	plot_accuracy(model, "Gaussian Naive Bayes",7,plot)
plot_roc(model, "Gaussian Naive Bayes",7)

# ----------- Neural network - Multi-layer Perceptron  ------------

model = MLPClassifier()
model = MLPClassifier(hidden_layer_sizes= 100, activation='relu')
run_model(model, "MLP Neural network")
for plot in range(num_plots):
	plot_accuracy(model,"MLP Neural network",8,plot)
plot_roc(model, "MLP Neural network",8)

#----------- LogisticRegression ------------

model = LogisticRegression(penalty='l1', tol=0.01) 
run_model(model, "LogisticRegression")
for plot in range(num_plots):
	plot_accuracy(model,"Logistic Regression",9,plot)
plot_roc(model, "LogisticRegression",9)

#----------- GradientBoostingClassifier -----------

model = GradientBoostingClassifier(learning_rate=0.01, max_depth=5, n_estimators=200)
run_model(model, "GradientBoostingClassifier")
for plot in range(num_plots):
	plot_accuracy(model,"GradientBoostingClassifier",10,plot)
plot_roc(model, "GradienBosstingClassifier",10)

for plot in range(num_plots):
	fig[plot].savefig("Classification_NoMC_"+removedVars+"_"+feature_labels[2*plot]+"_"+feature_labels[2*plot+1]+".png")

fig_roc.savefig("ROCcurve_"+removedVars+".png")

