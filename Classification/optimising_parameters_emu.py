import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier


use_lappd_info = 0    #should the model use data from the LAPPDs? 1: yes, 0: no

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
        X = balanced_data.iloc[:,[0,1,2,3,4,5,8,9,12,13,14,15,17,19,20,21,33,34,35,36,37,38,39,40]]
else:
        X = balanced_data.iloc[:,[0,1,2,3,4,5,8,9,12,13,14,15,17,19,20,21,22,23,25,26,29,30,33,34,35,36,37,38,39,40]]
y = balanced_data.iloc[:,46:47]  # Classification on the particle type

#specify in string which variables are not used
removedVars = "Beamlike_FV_PMTVol_DigitThr10"

print("X_data: ",X)
print("Y_data: ",y)

feature_labels=list(X.columns)


# ----- Build train and test dataset ---------
X_train0, X_test0, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 100)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train0))
X_test = pd.DataFrame(scaler.transform(X_test0))
print("type(X_train) ",type(X_train)," type(X_train0): ",type(X_train0))
y_train2 = np.array(y_train).ravel() #Return a contiguous flattened 1-d array
print("X_train.shape: ",X_train.shape," y_train2.shape: ",y_train2.shape, "X_test.shape: ",X_test.shape," y_test.shape: ",y_test.shape)

print('y_train2 before replacing: ',y_train2)
#make everything binary
y_train = np.where(y_train == 'muon', 0, 1)
y_train2 = np.where(y_train2 == 'muon', 0, 1)
y_test = np.where(y_test == 'muon', 0, 1)
print('y_train2 after replacing: ',y_train2)

# ------ optimise model ------ 
def optimise_model(model, alg_name, search_method, params):
    folds = 3
    param_comb = 5
    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

    #random search is faster than GridSearch but GridSearch is more detailed
    #(see https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost for details): 
    if search_method=="random":
       #---- RandomizedSearchCV ----
       random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train, y_train2), verbose=3, random_state=1001 )
    if search_method=="grid":
       #---- GridSearchCV ----
       random_search = GridSearchCV(model, param_grid=params, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train, y_train2), verbose=3)

    random_search.fit(X_train, y_train2)

    print("------ Algorithm: " + str(alg_name))
    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
    print(random_search.best_score_ * 2 - 1)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)
    print("----------------------------------------------")

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

# A parameter grid for GradientBoostingClassifier
params = {
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "max_depth":[3, 5, 8, 10],
    "n_estimators":[10, 100, 200, 500, 600]
    } 

model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)
optimise_model(model, " GradientBoostingClassifier ", "random", params)


