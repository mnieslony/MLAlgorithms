import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier


use_lappd_info = false

#------- Merge .csv files -------
data_e = pd.read_csv("data/beamlike_electron_FV_PMTVol_0_277.csv", header = 0)
data_e[46] = "electron"
data_mu = pd.read_csv("data/beam_muon_FV_PMTVol_0_4996.csv", header = 0)
data_mu[46] = "muon"
data = pd.concat([data_e,data_mu],axis=0)

# ------ Load data -----------
if not use_lappd_info:
    X = data.iloc[:,[0,1,2,3,4,5,8,9,12,13,14,15,17,19,20,21,33,34,35,36,37,38,39,40,43]]
else:
    X = data.iloc[:,[0,1,2,3,4,5,8,9,12,13,14,15,17,19,20,21,22,23,25,26,29,30,33,34,35,36,37,38,39,40,43]]
Y = data.iloc[:,46:47]  # Classification on the particle type

print("X_data: ",X)
print("Y_data: ",y)

# build train and test dataset
X_train0, X_test0, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 100)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train0))
X_test = pd.DataFrame(scaler.transform(X_test0))
print("type(X_train) ",type(X_train)," type(X_train0): ",type(X_train0))
y_train2 = np.array(y_train).ravel() #Return a contiguous flattened 1-d array
print("X_train.shape: ",X_train.shape," y_train2.shape: ",y_train2.shape, "X_test.shape: ",X_test.shape," y_test.shape: ",y_test.shape)

#-------------------------------------------------------
#------Feature Importance with ExtraTreesClassifier-----
#-------------------------------------------------------

print("-------------------------------------")
print("--------ExtraTrees Classifier--------")
print("-------------------------------------")

forest = ExtraTreesClassifier(n_estimators=250,random_state=0)
forest.fit(X_train,y_train2)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
print(importances)
print(indices)

feature_labels=list(X.columns)
feature_labels_sorted = [0] * len(feature_labels)

#print the feature ranking (Extra Trees Classifier)
print("ExtraTreesClassifier: ")
print("Feature ranking: ")
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    feature_labels_sorted[f] = feature_labels[indices[f]]

#plot the feature ranking (ExtraTrees Classifier)
plt.figure()
plt.title("Feature importances - ExtraTreesClassifier")
plt.bar(range(X_train.shape[1]), importances[indices],color="b",yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), feature_labels_sorted, rotation=45, fontsize=4)
plt.xlim([-1, X_train.shape[1]])
plt.ylabel("Importance")
plt.savefig("FeatureImportances_PID_ExtraTreesClassifier.pdf")


#-------------------------------------------------------
#-----Feature Importance with Univariate Selection------
#-------------------------------------------------------

print("-------------------------------------")
print("--------Univariate Selection---------")
print("-------------------------------------")

kBest = SelectKBest(score_func=f_classif, k=5)
fit_kBest = kBest.fit(X_train,y_train2)
np.set_printoptions(precision=3)
print("Univariate Selection:")
print(fit_kBest.scores_)
indices=np.argsort(fit_kBest.scores_)[::-1]
print(indices)
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], fit_kBest.scores_[indices[f]]))
    feature_labels_sorted[f] = feature_labels[indices[f]]
print(feature_labels_sorted)

plt.figure()
plt.title("Feature importances - KBest Univariate Selection")
plt.bar(range(X_train.shape[1]), fit_kBest.scores_[indices],color="b", align="center")
plt.xticks(range(X_train.shape[1]), feature_labels_sorted, rotation=45, fontsize=4)
plt.xlim([-1, X_train.shape[1]])
plt.ylabel("Importance")
plt.savefig("FeatureImportances_PID_UnivariateSelection.pdf")

#------Principal Component Analysis------------------------
#from sklearn.decomposition import PCA
# not really sure how to interprete reduced data --> omit for now

#pca = PCA(n_components=5)
#fit_pca = pca.fit(X_train)
#print("Principal Component Analysis:")
#print(("Explained Variance: %s") % fit_pca.explained_variance_ratio_)
#print(fit_pca.components_)


#---------------------------------------------------
#-----Recursive Feature Elimination (RFE)-----------
#---------------------------------------------------

# Test the ranking of variables for different models:

if not use_lappd_info:
    importancesAU = [18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
else:
    importancesAU = [24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]

print("-------------------------------------")
print("---Recursive Feature Elimination-----")
print("-------------------------------------")

# ----- Random Forest ---------------

model = RandomForestClassifier(n_estimators=100)
rfe_random = RFE(model,1)
rfe_random.fit(X_train,y_train2)

print("Random Forest:")
print(rfe_random.support_)
print(rfe_random.ranking_)

for f in range(X_train.shape[1]):
    index = np.where(rfe_random.ranking_==f+1)[0][0]
    print(index)
    feature_labels_sorted[f] = feature_labels[index]

print(feature_labels_sorted)

plt.figure()
plt.title("Feature importances - Random Forest")
plt.bar(range(X_train.shape[1]), importancesAU,color="b", align="center")
plt.xticks(range(X_train.shape[1]), feature_labels_sorted, rotation=45, fontsize=4)
plt.xlim([-1, X_train.shape[1]])
plt.ylabel("Importance [A.U.]")
plt.savefig("FeatureImportances_PID_RandomForest.pdf")

# ----- xgboost ------------

model = XGBClassifier(subsample=0.6, n_estimators=100, min_child_weight=5, max_depth=4, learning_rate=0.15, gamma=0.5, colsample_bytree=1.0)
rfe_xgb = RFE(model,1)
rfe_xgb.fit(X_train,y_train2)

print("XGB classifier:")
print(rfe_xgb.support_)
print(rfe_xgb.ranking_)

for f in range(X_train.shape[1]):
    index = np.where(rfe_xgb.ranking_==f+1)[0][0]
    print(index)
    feature_labels_sorted[f] = feature_labels[index]

print(feature_labels_sorted)

plt.figure()
plt.title("Feature importances - XGB")
plt.bar(range(X_train.shape[1]), importancesAU,color="b", align="center")
plt.xticks(range(X_train.shape[1]), feature_labels_sorted, rotation=45, fontsize=4)
plt.xlim([-1, X_train.shape[1]])
plt.ylabel("Importance [A.U.]")
plt.savefig("FeatureImportances_PID_XGB.pdf")

# ------ SVM Classifier ----------------
#gives error message: The classifier does not expose "coef_" or "feature_importances_" attributes (if not specifying kernel="linear")

model = SVC(probability=True, kernel="linear")
rfe_svc = RFE(model,1)
rfe_svc.fit(X_train,y_train2)

print("SVC classifier:")
print(rfe_svc.support_)
print(rfe_svc.ranking_)

for f in range(X_train.shape[1]):
    index = np.where(rfe_svc.ranking_==f+1)[0][0]
    print(index)
    feature_labels_sorted[f] = feature_labels[index]

print(feature_labels_sorted)

plt.figure()
plt.title("Feature importances - SVM")
plt.bar(range(X_train.shape[1]), importancesAU,color="b", align="center")
plt.xticks(range(X_train.shape[1]), feature_labels_sorted, rotation=45, fontsize=4)
plt.xlim([-1, X_train.shape[1]])
plt.ylabel("Importance [A.U.]")
plt.savefig("FeatureImportances_PID_SVM.pdf")


#----------- GradientBoostingClassifier -----------

model = GradientBoostingClassifier(learning_rate=0.01, max_depth=8, n_estimators=100)
rfe_gradientboosting = RFE(model,1)
rfe_gradientboosting.fit(X_train,y_train2)

print("Gradient Boosting classifier:")
print(rfe_gradientboosting.support_)
print(rfe_gradientboosting.ranking_)

for f in range(X_train.shape[1]):
    index = np.where(rfe_gradientboosting.ranking_==f+1)[0][0]
    print(index)
    feature_labels_sorted[f] = feature_labels[index]

print(feature_labels_sorted)

plt.figure()
plt.title("Feature importances - Gradient Boosting")
plt.bar(range(X_train.shape[1]), importancesAU,color="b", align="center")
plt.xticks(range(X_train.shape[1]), feature_labels_sorted, rotation=45, fontsize=4)
plt.xlim([-1, X_train.shape[1]])
plt.ylabel("Importance [A.U.]")
plt.savefig("FeatureImportances_PID_GradientBoosting.pdf")

