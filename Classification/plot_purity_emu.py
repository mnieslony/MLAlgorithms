import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


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

import pickle #for loading/saving models

import argparse #For user input


#------- Parse user arguments ----

parser = argparse.ArgumentParser(description='PID Plot purity/efficiency curves - Overview')
parser.add_argument("--input_e", default="data.nosync/beamlike_electron_DigitThr10_0_276_Full.csv", help = "The input electron file containing the data to be evaluated [csv-format]")
parser.add_argument("--input_mu", default="data.nosync/beamlike_muon_Digitthr10_0_498_Full.csv", help = "The input muon file containing the data to be evaluated [csv-format]")
parser.add_argument("--variable_names", default="VariableConfig_Full.txt", help = "File containing the list of classification variables")
parser.add_argument("--dataset_name",default="beamlike", help = "Keyword describing dataset name (used to label output files)")
parser.add_argument("--model_name",default="MLP",help="Classification model name. Options: RandomForest, XGBoost, SVM, SGD, MLP, GradientBoosting, All")
parser.add_argument("--frac_electron",default=0.01,help="Fraction of electron events in the beam sample.")

args = parser.parse_args()
input_file_e = args.input_e
input_file_mu = args.input_mu
variable_file = args.variable_names
dataset_name = args.dataset_name
model_name = args.model_name
frac_electron = args.frac_electron

balance_data = True     # Needs to be true, otherwise the weighted calculations are going to be wrong

print('PID Purity/Efficiency curves initialization: Electron data set: '+input_file_e+', muon data set: '+input_file_mu+', variable file: '+variable_file+', model: '+model_name)


#------- Merge .csv files -------

data_e = pd.read_csv(input_file_e,header = 0)
data_e['particleType'] = "electron"
data_mu = pd.read_csv(input_file_mu,header = 0)
data_mu['particleType'] = "muon"
data = pd.concat([data_e,data_mu],axis=0, ignore_index = True)    #ignore_index: one continuous index variable instead of separate ones for the 2 datasets


#------- Balance data (if specified) ----------

if balance_data:
    # Balance data to be 50% electron, 50% muon
    balanced_data = data.groupby('particleType')
    balanced_data = (balanced_data.apply(lambda x: x.sample(balanced_data.size().min()).reset_index(drop=True)))
else:
    balanced_data = data

#---------- Load only specific variables of data ---------

with open(variable_file) as f:
    subset_variables = f.read().splitlines()
subset_variables.append('particleType')

balanced_data = balanced_data[subset_variables]

variable_config = variable_file[15:variable_file.find('.txt')]
print('Variable configuration: ',variable_config)

# ---------- Fill x & y vectors with data ----------------

X = balanced_data.iloc[:,balanced_data.columns!='particleType']
y = balanced_data.iloc[:,-1]  # Classification on the particle type
print("Preview X (classification variables): ",X.head())
print("Preview Y (class names): ",y.head())

# Build train and test dataset
X_train0, X_test0, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 100)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train0))
X_test = pd.DataFrame(scaler.transform(X_test0))
print("type(X_train) ",type(X_train)," type(X_train0): ",type(X_train0))
y_train2 = np.array(y_train).ravel() # Return a continuous flattened 1-d array
print("X_train.shape: ",X_train.shape," y_train2.shape: ",y_train2.shape, "X_test.shape: ",X_test.shape," y_test.shape: ",y_test.shape)

# ------------------------------------------------------------------
# ----------- calculate_eff: Calculate efficiency ------------------
# ------------------------------------------------------------------

def calculate_eff(lower_bin,max_bins,out):
    return sum(out[lower_bin:max_bins])/(sum(out[0:max_bins]))

# ------------------------------------------------------------------
# ------------ calculate_purity: Calculate purity ------------------
# ------------------------------------------------------------------

def calculate_purity(lower_bin,max_bins,out1,out2,weighted,frac_minority):
    if weighted:
        return (1-frac_minority)*calculate_accepted(lower_bin,max_bins,out1)/((1-frac_minority)*calculate_accepted(lower_bin,max_bins,out1)+frac_minority*calculate_accepted(lower_bin,max_bins,out2))
    else:
        return calculate_accepted(lower_bin,max_bins,out1)/(calculate_accepted(lower_bin,max_bins,out1)+calculate_accepted(lower_bin,max_bins,out2))

# ----------------------------------------------------------------------------------------------
# ------------ calculate_relative_uncertainty: Calculate relative uncertainty ------------------
# ----------------------------------------------------------------------------------------------

def calculate_relative_uncertainty(lower_bin,max_bins,out1,out2,weighted,frac_minority):
    if weighted:
        return np.sqrt(np.power(1-frac_minority,2)*calculate_accepted(lower_bin,max_bins,out1)+2*np.power(frac_minority,2)*calculate_accepted(lower_bin,max_bins,out2))/((1-frac_minority)*calculate_accepted(lower_bin,max_bins,out1))
    else:
        return np.sqrt(calculate_accepted(lower_bin,max_bins,out1)+2*calculate_accepted(lower_bin,max_bins,out2))/calculate_accepted(lower_bin,max_bins,out1)

# ------------------------------------------------------------------
# -------- calculate_accepted: Calculate # events passing cut ------
# ------------------------------------------------------------------

def calculate_accepted(lower_bin,max_bins,out):
    return sum(out[lower_bin:max_bins])

# ------------------------------------------------------------------
# ---- run_model: Train the classifier on the training data set ----
# ------------------------------------------------------------------

def run_model(model, model_name):
    
    model.fit(X_train, y_train2)
    y_pred = model.predict(X_test)        
    report = classification_report(y_test, y_pred) 
    print(report)

# ------------------------------------------------------------------
# ---- plot_purity: Plot efficiency/purity/goodness curves----------
# ------------------------------------------------------------------
    
def plot_purity(model, model_name):

    y_pred = model.predict(X_test)
    accuracy =  accuracy_score(y_test, y_pred) * 100

    # Get predicted probability values
    y_pred_prob = model.predict_proba(X_test)
    #y_pred_logprob = model.predict_log_proba(X_test)
            
    proba_muon = []
    proba_electron = []
    proba_electron_muon = []
    proba_electron_electron = []
    i=0

    # Assign probabilities to true muon/electron classes
    for prob in y_pred_prob:
        if y_test.iloc[i]=="muon":
            proba_muon.append(prob[1])
            proba_electron_muon.append(prob[0])
        elif y_test.iloc[i]=="electron":
            proba_electron.append(prob[1])
            proba_electron_electron.append(prob[0])
        i=i+1

    # Plot predicted probability distributions
    out_muon = plt.hist(proba_muon,bins=101,range=(0,1),label='true = muon')
    out_electron = plt.hist(proba_electron,bins=101,alpha=0.5,range=(0,1),label='true = electron')
    plt.xlabel("pred probability")
    plt.title(model_name+" Prediction = Muon")
    plt.legend()
    plt.savefig("plots/PID/PredProbability/"+model_name+"_probMuon.pdf",format="pdf")
    plt.clf()

    out_electron_muon = plt.hist(proba_electron_muon,bins=101,range=(0,1),label='true = muon')
    out_electron_electron = plt.hist(proba_electron_electron,bins=101,alpha=0.5, range=(0,1),label='true = electron')
    plt.clf()

    # Print reference values at default cut of 50%
    print("Efficiency muon (50%): ",calculate_eff(50,101,out_muon[0]))
    print("Purity muon (50%): ",calculate_purity(50,101,out_muon[0],out_electron[0],False,frac_electron))
    print("Efficiency electron (50%): ",calculate_eff(50,101,out_electron_electron[0]))
    print("Purity electron (50%): ",calculate_purity(50,101,out_electron_electron[0],out_electron_muon[0],False,frac_electron))

    # Plot efficiency/purity/eff*purity/eff*purity^2 curves
    eff_muon = []
    purity_muon = []
    purity_weighted_muon = []
    eff_purity_muon = []
    goodness_muon = []
    eff_purity_weighted_muon = []
    goodness_weighted_muon = []
    rel_uncertainty_muon = []
    rel_uncertainty_weighted_muon = []

    for bin in range(101):
        temp_eff = calculate_eff(bin,101,out_muon[0])
        temp_purity = calculate_purity(bin,101,out_muon[0],out_electron[0],False,frac_electron)
        temp_purity_weighted = calculate_purity(bin,101,out_muon[0],out_electron[0],True,frac_electron)
        temp_rel_uncertainty = calculate_relative_uncertainty(bin,101,out_muon[0],out_electron[0],False,frac_electron)
        temp_rel_uncertainty_weighted = calculate_relative_uncertainty(bin,101,out_muon[0],out_electron[0],True,frac_electron)
        accepted_muon = calculate_accepted(bin,101,out_muon[0])
        accepted_electron = calculate_accepted(bin,101,out_electron[0])
        eff_muon.append(temp_eff)
        if abs(accepted_muon)<0.1 and abs(accepted_electron)<0.1:
            purity_muon.append(0)
            purity_weighted_muon.append(0)
        else:
            purity_muon.append(temp_purity)
            purity_weighted_muon.append(temp_purity_weighted)
        eff_purity_muon.append(temp_eff*temp_purity)
        goodness_muon.append(temp_eff*temp_purity*temp_purity)
        eff_purity_weighted_muon.append(temp_eff*temp_purity_weighted)
        goodness_weighted_muon.append(temp_eff*temp_purity_weighted*temp_purity_weighted)
        rel_uncertainty_muon.append(temp_rel_uncertainty)
        rel_uncertainty_weighted_muon.append(temp_rel_uncertainty_weighted)

    # Plot efficiency/purity curves (unweighted)

    x_values = np.arange(0,1.01,0.01)
    plt.plot(x_values,goodness_muon,label='$\mathregular{purity^2}$*efficiency',linestyle='-')
    plt.plot(x_values,eff_purity_muon,color='purple',label='purity*efficiency',linestyle='-')
    plt.plot(x_values,eff_muon,color='red',label='efficiency',linestyle='-')
    plt.plot(x_values,purity_muon,color='black',label='purity',linestyle='-')
    plt.title(model_name+" PID")
    plt.xlabel("pred probability (muon)")
    plt.legend()
    plt.savefig("plots/PID/PredProbability/"+model_name+"_PID_effpurity.pdf",format="pdf")
    plt.clf()

    # Plot efficiency/purity curves (weighted)

    plt.plot(x_values,goodness_weighted_muon,label='$\mathregular{purity^2}$*efficiency',linestyle='-')
    plt.plot(x_values,eff_purity_weighted_muon,color='purple',label='purity*efficiency',linestyle='-')
    plt.plot(x_values,eff_muon,color='red',label='efficiency',linestyle='-')
    plt.plot(x_values,purity_weighted_muon,color='black',label='purity',linestyle='-')
    plt.title(model_name+" PID (weighted)")
    plt.xlabel("pred probability (muon)")
    plt.legend()
    plt.savefig("plots/PID/PredProbability/"+model_name+"_PID_effpurity_weighted.pdf",format="pdf")
    plt.clf()

    # Plot relative uncertainty curves (unweighted)

    #plt.plot(x_values,eff_muon,color='red',label='efficiency',linestyle='-')
    plt.plot(x_values,rel_uncertainty_muon,color='black',label='rel uncertainty',linestyle='-')
    plt.title(model_name+" PID")
    plt.xlabel("pred probability (muon)")
    plt.legend()
    plt.savefig("plots/PID/PredProbability/"+model_name+"_PID_reluncert.pdf",format="pdf")
    plt.clf()
 
    # Plot relative uncertainty curves (weighted)

    #plt.plot(x_values,eff_muon,color='red',label='efficiency',linestyle='-')
    plt.plot(x_values,rel_uncertainty_weighted_muon,color='black',label='rel uncertainty',linestyle='-')
    plt.title(model_name+" PID")
    plt.xlabel("pred probability (muon)")
    plt.legend()
    plt.savefig("plots/PID/PredProbability/"+model_name+"_PID_reluncert_weighted.pdf",format="pdf")
    plt.clf()



# --------------------------------------------------------------------
# ------ eval_purity: Evaluate efficiency/purity/goodness data--------
# --------------------------------------------------------------------


def eval_purity(model,model_name,icol,eff_model,purity_model,eff_purity_model,goodness_model,reluncert_model,weighted):

    y_pred = model.predict(X_test)
    accuracy =  accuracy_score(y_test, y_pred) * 100 

    # Get predicted probability values
    y_pred_prob = model.predict_proba(X_test)
    #y_pred_logprob = model.predict_log_proba(X_test)

    proba_muon = []
    proba_electron = []
    proba_electron_muon = []
    proba_electron_electron = []
    i=0

    # Assign probabilities to true muon/electron classes
    for prob in y_pred_prob:

        if y_test.iloc[i]=="muon":
            proba_muon.append(prob[1])
            proba_electron_muon.append(prob[0])
        elif y_test.iloc[i]=="electron":
            proba_electron.append(prob[1])
            proba_electron_electron.append(prob[0])
        i=i+1

    # Plot predicted probability distributions
    out_muon = plt.hist(proba_muon,bins=101,range=(0,1),label='true = muon')
    out_electron = plt.hist(proba_electron,bins=101,alpha=0.5,range=(0,1),label='true = electron')
    plt.xlabel("pred probability")
    plt.title(model_name+" Prediction = Muon")
    plt.legend()
    plt.clf()

    # Save efficiency/purity/eff*purity/eff*purity^2 values to lists        
    eff_muon = []
    purity_muon = []
    eff_purity_muon = []
    goodness_muon = []
    reluncert_muon = []

    for bin in range(101):
        temp_eff = calculate_eff(bin,101,out_muon[0])
        temp_purity = calculate_purity(bin,101,out_muon[0],out_electron[0],weighted,frac_electron)
        temp_reluncert = calculate_relative_uncertainty(bin,101,out_muon[0],out_electron[0],weighted,frac_electron)
        accepted_muon = calculate_accepted(bin,101,out_muon[0])
        accepted_electron = calculate_accepted(bin,101,out_electron[0])
        if (abs(accepted_muon)<0.1) and (abs(accepted_electron)<0.1):
            eff_muon.append(0)
            purity_muon.append(0)
            eff_purity_muon.append(0)
            goodness_muon.append(0)
            reluncert_muon.append(temp_reluncert)
        else:
            eff_muon.append(temp_eff)
            purity_muon.append(temp_purity)
            reluncert_muon.append(temp_reluncert)
            eff_purity_muon.append(temp_eff*temp_purity)
            goodness_muon.append(temp_eff*temp_purity*temp_purity)


    for i in range(len(eff_muon)):
        eff_model.append(eff_muon[i])
        purity_model.append(purity_muon[i])
        eff_purity_model.append(eff_purity_muon[i])
        goodness_model.append(goodness_muon[i])
        reluncert_model.append(reluncert_muon[i])

    

# ----- Evaluate models with respect to accuracy, purity, efficiency, make plots ------------

if model_name == "All":
    model_names = ["RandomForest","MLP","XGB","SVM","SGD","GradientBoosting"]
    model_colors = ['blue','black','red','green','purple','grey']
else:
    model_names = [model_name]
    model_colors = ['blue']

eff_models=[]
purity_models=[]
purity_weighted_models=[]
eff_purity_models=[]
eff_purity_weighted_models=[]
goodness_models=[]
goodness_weighted_models=[]
reluncert_models=[]
reluncert_weighted_models=[]

# Plot pred comparison accuracy/purity curves for all specified classifiers

for imodel in range(len(model_names)):

    print('  ')
    print('///////////////////////////////////')
    print('   //////',model_names[imodel],'///////   ')
    print('///////////////////////////////////')
    print('  ')

    eff_model=[]
    purity_model=[]
    purity_weighted_model=[]
    eff_purity_model=[]
    eff_purity_weighted_model=[]
    goodness_model=[]
    goodness_weighted_model=[]
    reluncert_model=[]
    reluncert_weighted_model=[]

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
    eval_purity(model,model_names[imodel],model_colors[imodel],eff_model,purity_model,eff_purity_model,goodness_model,reluncert_model,False)
    eff_model=[]
    eval_purity(model,model_names[imodel],model_colors[imodel],eff_model,purity_weighted_model,eff_purity_weighted_model,goodness_weighted_model,reluncert_weighted_model,True)
    eff_models.append(eff_model)
    purity_models.append(purity_model)
    purity_weighted_models.append(purity_weighted_model)
    eff_purity_models.append(eff_purity_model)
    eff_purity_weighted_models.append(eff_purity_weighted_model)
    goodness_models.append(goodness_model)
    goodness_weighted_models.append(goodness_weighted_model)
    reluncert_models.append(reluncert_model)
    reluncert_weighted_models.append(reluncert_weighted_model)
    plot_purity(model,model_names[imodel])


# Plot efficiency curves on top of each other

x_values = np.arange(0,1.01,0.01)
for imodel in range(len(model_names)):
    plt.plot(x_values,eff_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("efficiency")
plt.title("Efficiency comparison - PID")
plt.legend()
plt.savefig("plots/PID/PredProbability/PID_EffComparison_"+dataset_name+"_"+variable_config+".pdf",format="pdf")
plt.clf()

# Plot purity curves on top of each other

for imodel in range(len(model_names)):
    plt.plot(x_values,purity_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("purity")
plt.title("Purity comparison - PID")
plt.legend()
plt.savefig("plots/PID/PredProbability/PID_PurityComparison_"+dataset_name+"_"+variable_config+".pdf",format="pdf")
plt.clf()

for imodel in range(len(model_names)):
    plt.plot(x_values,purity_weighted_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("purity")
plt.title("Purity comparison - PID (weighted)")
plt.legend()
plt.savefig("plots/PID/PredProbability/PID_PurityComparison_weighted_"+dataset_name+"_"+variable_config+".pdf",format="pdf")
plt.clf()

# Plot efficiency*purity curves on top of each other

for imodel in range(len(model_names)):
    plt.plot(x_values,eff_purity_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("$efficiency*purity$")
plt.title("$Efficiency*Purity$ comparison - PID")
plt.legend()
plt.savefig("plots/PID/PredProbability/PID_EffPurityComparison_"+dataset_name+"_"+variable_config+".pdf",format="pdf")
plt.clf()

for imodel in range(len(model_names)):
    plt.plot(x_values,eff_purity_weighted_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("$efficiency*purity$")
plt.title("$Efficiency*Purity$ comparison - PID (weighted)")
plt.legend()
plt.savefig("plots/PID/PredProbability/PID_EffPurityComparison_weighted_"+dataset_name+"_"+variable_config+".pdf",format="pdf")
plt.clf()


# Plot efficiency*purity^2 curves on top of each other

for imodel in range(len(model_names)):
    plt.plot(x_values,goodness_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("$efficiency*purity^2$")
plt.title("$Efficiency*Purity^2$ comparison - PID")
plt.legend()
plt.savefig("plots/PID/PredProbability/PID_EffPurity2Comparison_"+dataset_name+"_"+variable_config+".pdf",format="pdf")
plt.clf()

for imodel in range(len(model_names)):
    plt.plot(x_values,goodness_weighted_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("$efficiency*purity^2$")
plt.title("$Efficiency*Purity^2$ comparison - PID")
plt.legend()
plt.savefig("plots/PID/PredProbability/PID_EffPurity2Comparison_weighted_"+dataset_name+"_"+variable_config+".pdf",format="pdf")
plt.clf()

# Plot relative uncertainty curves on top of each other

for imodel in range(len(model_names)):
    plt.plot(x_values,reluncert_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("$\sqrt{S+2B}/S$")
plt.title("Rel. uncertainty comparison - PID")
plt.legend()
plt.savefig("plots/PID/PredProbability/PID_RelUncertainty_"+dataset_name+"_"+variable_config+".pdf",format="pdf")
plt.clf()

for imodel in range(len(model_names)):
    plt.plot(x_values,reluncert_weighted_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("$\sqrt{S+2B}/S$")
plt.title("Rel. uncertainty comparison - PID")
plt.legend()
plt.savefig("plots/PID/PredProbability/PID_RelUncertainty_weighted_"+dataset_name+"_"+variable_config+".pdf",format="pdf")
plt.clf()



