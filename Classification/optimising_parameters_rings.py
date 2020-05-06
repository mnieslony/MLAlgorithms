import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

import argparse

#------- Parse user arguments ----

parser = argparse.ArgumentParser(description='Ring Classification Optimise parameters - Overview')
parser.add_argument("--input", default="data.nosync/beam_muon_FV_PMTVol_SingleMultiRing_DigitThr10_wPhi_0_4996.csv", help = "The input single/multi-ring file containing the data to be evaluated [csv-format]")
parser.add_argument("--balance_data",default=True,help="Should the two classes in the dataset be balanced 50/50?")
parser.add_argument("--variable_names", default="VariableConfig_Old.txt", help = "File containing the list of classification variables")
parser.add_argument("--dataset_name",default="beam", help = "Keyword describing dataset name (used to label output files)")

args = parser.parse_args()
input_file = args.input
balance_data = args.balance_data
variable_file = args.variable_names
dataset_name = args.dataset_name

print('Ring Classification Optimise parameters - Initialisation: Input file: ',input_file,', variable file: ',variable_file)

#------- Load .csv files -------
data = pd.read_csv(input_file, header = 0)
data['multiplerings'] = data['multiplerings'].astype('str')
data.replace({'multiplerings':{'0':'1-ring',str(1):'multi-ring'}},inplace=True)


# ------ Balance data (if specified) -------

if balance_data:
    # Balance multi and single ring events (50% / 50%)
    balanced_data = data.groupby('multiplerings')
    balanced_data = (balanced_data.apply(lambda x: x.sample(balanced_data.size().min()).reset_index(drop=True)))
    print ("Multiple rings counts (after balancing): ",balanced_data['multiplerings'].value_counts())

else:
    balanced_data = data
    print ("Multiple rings counts (not balanced): ",balanced_data['multiplerings'].value_counts())

#---------- Load only specific variables of data ---------

with open(variable_file) as f:
    subset_variables = f.read().splitlines()
subset_variables.append('multiplerings')

balanced_data = balanced_data[subset_variables]

variable_config = variable_file[15:variable_file.find('.txt')]
print('Variable configuration: ',variable_config)


# ---------- Fill x & y vectors with data ----------------

X = balanced_data.iloc[:,balanced_data.columns!="multiplerings"]
y = balanced_data.iloc[:,-1]
print("Preview X (classification variables): ",X.head())
print("Preview Y (class names): ",y.head())

#X = balanced_data.iloc[:,[0,1,2,3,4,5,8,9,12,13,14,15,17,19,20,21,33,34]]   # ignore first column which is row Id
#y = balanced_data.iloc[:,42:43]  # Classification on the boolean 'multiplerings'

# ----- Build train and test dataset ---------

X_train0, X_test0, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 100)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train0))
X_test = pd.DataFrame(scaler.transform(X_test0))
print("type(X_train) ",type(X_train)," type(X_train0): ",type(X_train0))
y_train2 = np.array(y_train).ravel() #Return a contiguous flattened 1-d array
print("X_train.shape: ",X_train.shape," y_train2.shape: ",y_train2.shape, "X_test.shape: ",X_test.shape," y_test.shape: ",y_test.shape)

# Make everything binary
print('y_train2 before replacing: ',y_train2)
y_train = np.where(y_train == '1-ring', 0, 1)
y_train2 = np.where(y_train2 == '1-ring', 0, 1)
y_test = np.where(y_test == '1-ring', 0, 1)
print('y_train2 after replacing: ',y_train2)

# ------------------------------------------------------------------
# ---------- optimise_model: Evaluate through hyperparameters ------
# ------------------------------------------------------------------

file = open("hyperparameters/RingClassification/RingClassification_hyperparameter_scan_"+dataset_name+"_"+variable_config+".txt","w")

def optimise_model(model, alg_name, search_method, params):
    folds = 3
    param_comb = 5
    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

    #random search is faster than GridSearch but GridSearch is more detailed
    #(see https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost for details): 
    if search_method=="random":
       #---- RandomizedSearchCV ----
       random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train, y_train2), verbose=3, random_state=1001)
    if search_method=="grid":
       #---- GridSearchCV ----
       random_search = GridSearchCV(model, param_grid=params, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train, y_train2), verbose=3, pos_label='1-ring')

    random_search.fit(X_train, y_train2)

    print("------ Algorithm: " + str(alg_name))
    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
    print(random_search.best_score_ * 2 - 1)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)
    print("----------------------------------------------")

    file.write("\n Algorithm: " + str(alg_name))
    file.write('\n Best estimator: ')
    file.write(str(random_search.best_estimator_))
    file.write('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
    file.write(str(random_search.best_score_ * 2 - 1))
    file.write('\n Best hyperparameters:')
    file.write(str(random_search.best_params_))
    file.write("\n ----------------------------------------------")

# ------ Test optimal parameters for different models: --------
# ----- xgboost ------------

# A parameter grid for XGBoost
params = {
        "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
        "n_estimators":[10, 100, 200, 500, 600],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

model = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)
optimise_model(model, "XGBoost", "random", params)

# ----------- Neural network - Multi-layer Perceptron  ------------

# A parameter grid for MLP
params = {
        "hidden_layer_sizes": [10, 50, 100],
        "activation": ['relu', 'tanh']
        } 
model = MLPClassifier()
optimise_model(model, " MLP Neural network ", "random", params)

#----------- GradientBoostingClassifier -----------

# A parameter grid for GradientBoosting
params = {
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "max_depth":[3, 5, 8, 10],
    "n_estimators":[10, 100, 200, 500, 600]
    } 

model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)
optimise_model(model, " GradientBoostingClassifier ", "random", params)

file.close()

