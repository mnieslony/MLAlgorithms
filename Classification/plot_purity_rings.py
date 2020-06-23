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
from sklearn.model_selection import train_test_split

import pickle #for saving models

import argparse #For user input


#------- Parse user arguments ----

parser = argparse.ArgumentParser(description='Ring Classification Plot purity/efficiency curves - Overview')
parser.add_argument("--input", default="data_new.nosync/beam_DigitThr10_0_4996_Full.csv", help = "The input electron file containing the data to be evaluated [csv-format]")
parser.add_argument("--variable_names", default="VariableConfig_Full.txt", help = "File containing the list of classification variables")
parser.add_argument("--dataset_name",default="beam", help = "Keyword describing dataset name (used to label output files)")
parser.add_argument("--model_name",default="MLP",help="Classification model name. Options: RandomForest, XGBoost, SVM, SGD, MLP, GradientBoosting, All")
parser.add_argument("--frac_multi",default=0.37,help="Fraction of multi-ring events in the beam sample.")
parser.add_argument("--status_suffix",default="Full.csv",help="Name of the last portion of the data file.")

args = parser.parse_args()
input_file = args.input
variable_file = args.variable_names
dataset_name = args.dataset_name
model_name = args.model_name
frac_multi = args.frac_multi
status_suffix = args.status_suffix

balance_data = True     # Needs to be true, otherwise the weighted calculations are going to be wrong

print('Ring Classification Purity/Efficiency curves initialization: Data set: '+input_file+', variable file: '+variable_file+', model: '+model_name)


#------- Read in .csv file -------

data = pd.read_csv(input_file,header = 0)   #first row is header
input_file_additional = input_file[0:input_file.find(status_suffix)]+"status.csv"
data_additional = pd.read_csv(input_file_additional, header = 0)
data = pd.concat([data,data_additional], axis=1, sort = False)
data['MCMultiRing'] = data['MCMultiRing'].astype('str')
data.replace({'MCMultiRing':{'0':'1-ring',str(1):'multi-ring'}},inplace=True)

#--------- Balance data (if specified) ---------

if balance_data:
    balanced_data = data.groupby('MCMultiRing')
    balanced_data = (balanced_data.apply(lambda x: x.sample(balanced_data.size().min()).reset_index(drop=True)))
else:
    balanced_data = data

#---------- Load only specific variables of data ---------

variable_file_path = "variable_config/"+variable_file
with open(variable_file_path) as f:
    subset_variables = f.read().splitlines()
subset_variables.append('MCMultiRing')

balanced_data = balanced_data[subset_variables]

variable_config = variable_file[15:variable_file.find('.txt')]
print('Variable configuration: ',variable_config)

# ---------- Fill x & y vectors with data ----------------

X = balanced_data.iloc[:,balanced_data.columns!='MCMultiRing']
y = balanced_data.iloc[:,-1]  # Classification on the single-ring/multi-ring class
print("Preview X (classification variables): ",X.head())
print("Preview Y (class names): ",y.head())


# Build train and test dataset
X_train0, X_test0, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 100)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train0))
X_test = pd.DataFrame(scaler.transform(X_test0))
print("type(X_train) ",type(X_train)," type(X_train0): ",type(X_train0))
y_train2 = np.array(y_train).ravel() #Return a contiguous flattened 1-d array
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

    purity_file = open("plots/RingClassification/PredProbability/PurityValues_RingClassification_"+model_name+"_"+dataset_name+"_"+variable_config+".csv","w")
    purity_file_weighted = open("plots/RingClassification/PredProbability/PurityValuesWeighted_RingClassification_"+model_name+"_"+dataset_name+"_"+variable_config+".csv","w")
    purity_file.write("probability,efficiency,purity,rel_uncertainty,efficiency*purity,efficiency*purity**2\n")
    purity_file_weighted.write("probability,efficiency,purity,rel_uncertainty,efficiency*purity,efficiency*purity**2\n")

    y_pred = model.predict(X_test)
    accuracy =  accuracy_score(y_test, y_pred) * 100 

    # Get predicted probability values
    y_pred_prob = model.predict_proba(X_test)
    #y_pred_logprob = model.predict_log_proba(X_test)

    proba_singlering = []
    proba_multiring = []
    proba_multi_singlering = []
    proba_multi_multiring = []
    i=0

    # Assign probabilities to true single/multi-ring classes
    for prob in y_pred_prob:
        if y_test.iloc[i]=="1-ring":
            proba_singlering.append(prob[0])
            proba_multi_singlering.append(prob[1])
        elif y_test.iloc[i]=="multi-ring":
            proba_multiring.append(prob[0])
            proba_multi_multiring.append(prob[1])
        i=i+1

    print('length of proba_multiring array: ',len(proba_multiring))

    # Plot predicted probability distributions
    out_multi = plt.hist(proba_multiring,bins=101,range=(0,1),label='true = multi ring')
    out_single = plt.hist(proba_singlering,bins=101,alpha=0.5,range=(0,1),label='true = single ring')    
    plt.xlabel("pred probability")
    plt.title(model_name+" Prediction = SingleRing")
    plt.legend()
    plt.savefig("plots/RingClassification/PredProbability/"+model_name+"_"+dataset_name+"_"+variable_config+"_probSingleRing.pdf",format="pdf")
    
    plt.yscale("log")
    plt.savefig("plots/RingClassification/PredProbability/"+model_name+"_"+dataset_name+"_"+variable_config+"_probSingleRing_log.pdf",format="pdf")
    plt.clf()

    out_multi_single = plt.hist(proba_multi_singlering,bins=101,range=(0,1),label='true = single ring')
    out_multi_multi = plt.hist(proba_multi_multiring,bins=101,alpha=0.5, range=(0,1),label='true = multi ring')
    plt.clf()
    
    # Print reference values at default cut of 50%
    print("Efficiency single ring (50%): ",calculate_eff(50,101,out_single[0]))
    print("Efficiency multi rings (50%): ",calculate_eff(50,101,out_multi_multi[0]))
    print("Purity single ring (50%): ",calculate_purity(50,101,out_single[0],out_multi[0],False,frac_multi))
    print("Purity multi ring (50%): ",calculate_purity(50,101,out_multi_multi[0],out_multi_single[0],False,frac_multi))

    # Plot efficiency/purity/eff*purity/eff*purity^2 curves  
    eff_single = []
    purity_single = []
    purity_weighted_single = []
    eff_purity_single = []
    goodness_single = []
    eff_purity_weighted_single = []
    goodness_weighted_single = []
    rel_uncertainty_single = []
    rel_uncertainty_weighted_single = []

    for bin in range(101):
        temp_eff = calculate_eff(bin,101,out_single[0])
        temp_purity = calculate_purity(bin,101,out_single[0],out_multi[0],False,frac_multi)
        temp_purity_weighted = calculate_purity(bin,101,out_single[0],out_multi[0],True,frac_multi)
        temp_rel_uncertainty = calculate_relative_uncertainty(bin,101,out_single[0],out_multi[0],False,frac_multi)
        temp_rel_uncertainty_weighted = calculate_relative_uncertainty(bin,101,out_single[0],out_multi[0],True,frac_multi)
        accepted_single = calculate_accepted(bin,101,out_single[0])
        accepted_multi = calculate_accepted(bin,101,out_multi[0])
        eff_single.append(temp_eff)
        if abs(accepted_single)<0.1 and abs(accepted_multi)<0.1:
            purity_single.append(0)
            purity_weighted_single.append(0)
        else:
            purity_single.append(temp_purity)
            purity_weighted_single.append(temp_purity_weighted)
        eff_purity_single.append(temp_eff*temp_purity)
        goodness_single.append(temp_eff*temp_purity*temp_purity)
        eff_purity_weighted_single.append(temp_eff*temp_purity_weighted)
        goodness_weighted_single.append(temp_eff*temp_purity_weighted*temp_purity_weighted)
        rel_uncertainty_single.append(temp_rel_uncertainty)
        rel_uncertainty_weighted_single.append(temp_rel_uncertainty_weighted)
        purity_file.write(str(bin/100)+","+str(temp_eff)+","+str(temp_purity)+","+str(temp_rel_uncertainty)+","+str(temp_eff*temp_purity)+","+str(temp_eff*temp_purity*temp_purity)+"\n")
        purity_file_weighted.write(str(bin/100)+","+str(temp_eff)+","+str(temp_purity_weighted)+","+str(temp_rel_uncertainty_weighted)+","+str(temp_eff*temp_purity_weighted)+","+str(temp_eff*temp_purity_weighted*temp_purity_weighted)+"\n")

    purity_file.close()
    purity_file_weighted.close()

    # Plot efficiency/purity curves (unweighted)

    x_values = np.arange(0,1.01,0.01)
    plt.plot(x_values,goodness_single,label='$\mathregular{purity^2}$*efficiency',linestyle='-')
    plt.plot(x_values,eff_purity_single,label='purity*efficiency',linestyle='-')
    plt.plot(x_values,eff_single,color='red',label='efficiency',linestyle='-')
    plt.plot(x_values,purity_single,color='black',label='purity',linestyle='-')
    plt.title(model_name+" Ring Classification")
    plt.xlabel("pred probability")
    plt.ylabel("purity/efficiency")
    plt.legend()
    plt.savefig("plots/RingClassification/PredProbability/"+model_name+"_RingClassification_"+dataset_name+"_"+variable_config+"_effpurity.pdf",format="pdf")
    plt.clf()

    # Plot efficiency/purity curves (weighted)

    plt.plot(x_values,goodness_weighted_single,label='$\mathregular{purity^2}$*efficiency',linestyle='-')
    plt.plot(x_values,eff_purity_weighted_single,color='purple',label='purity*efficiency',linestyle='-')
    plt.plot(x_values,eff_single,color='red',label='efficiency',linestyle='-')
    plt.plot(x_values,purity_weighted_single,color='black',label='purity',linestyle='-')
    plt.title(model_name+" Ring Classification (weighted)")
    plt.xlabel("pred probability")
    plt.ylabel("purity/efficiency")
    plt.legend()
    plt.savefig("plots/RingClassification/PredProbability/"+model_name+"_RingClassification_"+dataset_name+"_"+variable_config+"_effpurity_weighted.pdf",format="pdf")
    plt.clf()

    # Plot relative uncertainty curves (unweighted)

    #plt.plot(x_values,eff_muon,color='red',label='efficiency',linestyle='-')
    plt.plot(x_values,rel_uncertainty_single,color='black',label='rel uncertainty',linestyle='-')
    plt.title(model_name+" Ring Classification")
    plt.xlabel("pred probability")
    plt.ylabel("$\sqrt{S+2B}/S$")
    plt.legend()
    plt.savefig("plots/RingClassification/PredProbability/"+model_name+"_RingClassification_"+dataset_name+"_"+variable_config+"_reluncert.pdf",format="pdf")
    plt.clf()
 
    # Plot relative uncertainty curves (weighted)

    #plt.plot(x_values,eff_muon,color='red',label='efficiency',linestyle='-')
    plt.plot(x_values,rel_uncertainty_weighted_single,color='black',label='rel uncertainty',linestyle='-')
    plt.title(model_name+" Ring Classification (weighted)")
    plt.xlabel("pred probability")
    plt.ylabel("$\sqrt{S+2B}/S$")
    plt.legend()
    plt.savefig("plots/RingClassification/PredProbability/"+model_name+"_RingClassification_"+dataset_name+"_"+variable_config+"_reluncert_weighted.pdf",format="pdf")
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

    proba_singlering = []
    proba_multiring = []
    proba_multi_singlering = []
    proba_multi_multiring = []
    i=0


    # Assign probabilities to true muon/electron classes
    for prob in y_pred_prob:

        if y_test.iloc[i]=="1-ring":
            proba_singlering.append(prob[0])
            proba_multi_singlering.append(prob[1])
        elif y_test.iloc[i]=="multi-ring":
            proba_multiring.append(prob[0])
            proba_multi_multiring.append(prob[1])
        i=i+1

    # Plot predicted probability distributions    
    out_single = plt.hist(proba_singlering,bins=101,range=(0,1),label='true = single ring')
    out_multi = plt.hist(proba_multiring,bins=101,range=(0,1),label='true = multi ring')
    plt.clf()

    # Save efficiency/purity/eff*purity/eff*purity^2 values to lists        
    eff_single = []
    purity_single = []
    eff_purity_single = []
    goodness_single = []
    reluncert_single = []

    for bin in range(101):

        temp_eff = calculate_eff(bin,101,out_single[0])
        temp_purity = calculate_purity(bin,101,out_single[0],out_multi[0],weighted,frac_multi)
        temp_reluncert = calculate_relative_uncertainty(bin,101,out_single[0],out_multi[0],weighted,frac_multi)
        accepted_single = calculate_accepted(bin,101,out_single[0])
        accepted_multi = calculate_accepted(bin,101,out_multi[0])
        if (abs(accepted_single)<0.1) and (abs(accepted_multi)<0.1):
            eff_single.append(0)
            purity_single.append(0)
            eff_purity_single.append(0)
            goodness_single.append(0)
            reluncert_single.append(temp_reluncert)
        else:
            eff_single.append(temp_eff)
            purity_single.append(temp_purity)
            reluncert_single.append(temp_reluncert)
            eff_purity_single.append(temp_eff*temp_purity)
            goodness_single.append(temp_eff*temp_purity*temp_purity)


    for i in range(len(eff_single)):
        eff_model.append(eff_single[i])
        purity_model.append(purity_single[i])
        eff_purity_model.append(eff_purity_single[i])
        goodness_model.append(goodness_single[i])
        reluncert_model.append(reluncert_single[i])
    

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

#Plot pred comparison accuracy/purity curves between all classifiers

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
plt.title("Efficiency comparison - Ring Classification")
plt.legend()
plt.savefig("plots/RingClassification/PredProbability/RingClassification_EffComparison_"+dataset_name+"_"+variable_config+".pdf",format="pdf")
plt.clf()

# Plot purity curves on top of each other

for imodel in range(len(model_names)):
    plt.plot(x_values,purity_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("purity")
plt.title("Purity comparison - Ring Classification")
plt.legend()
plt.savefig("plots/RingClassification/PredProbability/RingClassification_PurityComparison_"+dataset_name+"_"+variable_config+".pdf",format="pdf")
plt.clf()

for imodel in range(len(model_names)):
    plt.plot(x_values,purity_weighted_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("purity")
plt.title("Purity comparison - Ring Classification (weighted)")
plt.legend()
plt.savefig("plots/RingClassification/PredProbability/RingClassification_PurityComparison_weighted_"+dataset_name+"_"+variable_config+".pdf",format="pdf")
plt.clf()

# Plot efficiency*purity curves on top of each other

for imodel in range(len(model_names)):
    plt.plot(x_values,eff_purity_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("$efficiency*purity$")
plt.title("$Efficiency*Purity$ comparison - Ring Classification")
plt.legend()
plt.savefig("plots/RingClassification/PredProbability/RingClassification_EffPurityComparison_"+dataset_name+"_"+variable_config+".pdf",format="pdf")
plt.clf()

for imodel in range(len(model_names)):
    plt.plot(x_values,eff_purity_weighted_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("$efficiency*purity$")
plt.title("$Efficiency*Purity$ comparison - Ring Classification (weighted)")
plt.legend()
plt.savefig("plots/RingClassification/PredProbability/RingClassification_EffPurityComparison_weighted_"+dataset_name+"_"+variable_config+".pdf",format="pdf")
plt.clf()


# Plot efficiency*purity^2 curves on top of each other

for imodel in range(len(model_names)):
    plt.plot(x_values,goodness_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("$efficiency*purity^2$")
plt.title("$Efficiency*Purity^2$ comparison - Ring Classification")
plt.legend()
plt.savefig("plots/RingClassification/PredProbability/RingClassification_EffPurity2Comparison_"+dataset_name+"_"+variable_config+".pdf",format="pdf")
plt.clf()

for imodel in range(len(model_names)):
    plt.plot(x_values,goodness_weighted_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("$efficiency*purity^2$")
plt.title("$Efficiency*Purity^2$ comparison - Ring Classification")
plt.legend()
plt.savefig("plots/RingClassification/PredProbability/RingClassification_EffPurity2Comparison_weighted_"+dataset_name+"_"+variable_config+".pdf",format="pdf")
plt.clf()

# Plot relative uncertainty curves on top of each other

for imodel in range(len(model_names)):
    plt.plot(x_values,reluncert_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("$\sqrt{S+2B}/S$")
plt.title("Rel. uncertainty comparison - Ring Classification")
plt.legend()
plt.savefig("plots/RingClassification/PredProbability/RingClassification_RelUncertainty_"+dataset_name+"_"+variable_config+".pdf",format="pdf")
plt.clf()

for imodel in range(len(model_names)):
    plt.plot(x_values,reluncert_weighted_models[imodel],linestyle='-',color=model_colors[imodel],label=model_names[imodel])

plt.xlabel("pred probability")
plt.ylabel("$\sqrt{S+2B}/S$")
plt.title("Rel. uncertainty comparison - Ring Classification")
plt.legend()
plt.savefig("plots/RingClassification/PredProbability/RingClassification_RelUncertainty_weighted_"+dataset_name+"_"+variable_config+".pdf",format="pdf")
plt.clf()



