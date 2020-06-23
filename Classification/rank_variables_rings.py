import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



import argparse #For user input

#------- Parse user arguments ----

parser = argparse.ArgumentParser(description='RingClassification Rank variables - Overview')
parser.add_argument("--input", default="data.nosync/beam_muon_FV_PMTVol_SingleMultiRing_DigitThr10_wPhi_0_4996_Old.csv", help = "The input file containing the single/multi-ring data to be evaluated [csv-format]")
parser.add_argument("--balance_data",default=True,help="Should the two classes in the dataset be balanced 50/50?")
parser.add_argument("--variable_names", default="VariableConfig_Old.txt", help = "File containing the list of classification variables")
parser.add_argument("--dataset_name",default="beam", help = "Keyword describing dataset name (used to label output files)")

args = parser.parse_args()
input_file = args.input
balance_data = args.balance_data
variable_file = args.variable_names
dataset_name = args.dataset_name

print('RingClassification Rank variables initialization: Input file: '+input_file+', variable config file: '+variable_file)


#------- Read in .csv file -------

data = pd.read_csv(input_file,header = 0)   #first row is header
data['multiplerings'] = data['multiplerings'].astype('str')
data.replace({'multiplerings':{'0':'1-ring',str(1):'multi-ring'}},inplace=True)


#-------- Balance data set (if specified) --------

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

X = balanced_data.iloc[:,balanced_data.columns!='multiplerings']
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

feature_labels = list(X.columns)
feature_labels_sorted = [0] * len(feature_labels)

# Print the feature ranking (ExtraTreesClassifier)
print("Feature ranking: ")
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    feature_labels_sorted[f] = feature_labels[indices[f]]

# Plot the feature ranking (ExtraTreesClassifier)
plt.title("Feature importances - ExtraTreesClassifier")
plt.bar(range(X_train.shape[1]), importances[indices],color="b",yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), feature_labels_sorted, rotation=45, fontsize=4)
plt.xlim([-1, X_train.shape[1]])
plt.ylabel("Importance")
plt.savefig("plots/RingClassification/FeatureImportance/FeatureImportances_RingClassification_"+dataset_name+"_"+variable_config+"_ExtraTreesClassifier.pdf")

#-------------------------------------------------------
#-----Feature Importance with Univariate Selection------
#-------------------------------------------------------

print("-------------------------------------")
print("--------Univariate Selection---------")
print("-------------------------------------")

kBest = SelectKBest(score_func=f_classif, k=5)
fit_kBest = kBest.fit(X_train,y_train2)
np.set_printoptions(precision=3)
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
plt.savefig("plots/RingClassification/FeatureImportance/FeatureImportances_RingClassification_"+dataset_name+"_"+variable_config+"_UnivariateSelection.pdf")

#---------------------------------------------------
#------Principal Component Analysis (PCA)-----------
#---------------------------------------------------

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

importancesAU = []
for i in range(len(subset_variables)-1):
    importancesAU.append(len(subset_variables)-1-i)

print('Importances (A.U.), after initialization: ',importancesAU)

print("-------------------------------------")
print("---Recursive Feature Elimination-----")
print("-------------------------------------")

# ----- Random Forest ---------------

model = RandomForestClassifier(n_estimators=100)
rfe_random = RFE(model,1)
rfe_random.fit(X_train,y_train2)

print("///////////////////")
print("///Random Forest///")
print("///////////////////")
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
plt.savefig("plots/RingClassification/FeatureImportance/FeatureImportances_RingClassification_"+dataset_name+"_"+variable_config+"_RandomForest.pdf")

# ----- xgboost ------------

model = XGBClassifier(subsample=0.6, n_estimators=100, min_child_weight=5, max_depth=4, learning_rate=0.15, gamma=0.5, colsample_bytree=1.0)
rfe_xgb = RFE(model,1)
rfe_xgb.fit(X_train,y_train2)

print("////////////////////")
print("///XGB classifier///")
print("////////////////////")
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
plt.savefig("plots/RingClassification/FeatureImportance/FeatureImportances_RingClassification"+dataset_name+"_"+variable_config+"_XGB.pdf")

# ------ SVM Classifier ----------------
#gives error message: The classifier does not expose "coef_" or "feature_importances_" attributes (if not specifying kernel="linear")

model = SVC(probability=True, kernel="linear")
rfe_svc = RFE(model,1)
rfe_svc.fit(X_train,y_train2)

print("//////////////////////")
print("///SVC classifier/////")
print("/////////////////////")
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
plt.savefig("plots/RingClassification/FeatureImportance/FeatureImportances_RingClassification_"+dataset_name+"_"+variable_config+"_SVM.pdf")


#----------- GradientBoostingClassifier -----------
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(learning_rate=0.01, max_depth=8, n_estimators=100)
rfe_gradientboosting = RFE(model,1)
rfe_gradientboosting.fit(X_train,y_train2)

print("////////////////////////////////////")
print("///Gradient Boosting classifier/////")
print("////////////////////////////////////")
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
plt.savefig("plots/RingClassification/FeatureImportance/FeatureImportances_RingClassification_"+dataset_name+"_"+variable_config+"_GradientBoosting.pdf")

