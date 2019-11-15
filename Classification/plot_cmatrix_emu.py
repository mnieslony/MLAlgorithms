import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

class_types=["muon","electron"]

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          savepath="confusionmatrix.pdf"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    print(unique_labels(y_true,y_pred))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=class_types, yticklabels=class_types,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(savepath,format="pdf")
    return ax


np.set_printoptions(precision=2)

model_names=["RandomForest","MLP","XGBoost","SVM","SGD","GradientBoosting"]

#read classified data from speicific classifier
for model_name in model_names:
    data0 = pd.read_csv("PID_"+model_name+"_predictions_Beamlike_FV_PMTVol_DigitThr10.csv");
    print(data0.head())
    class_names = data0["particleType"].values
    print("class_names: ",class_names)
    #convert strings to numbers:
    data1 = data0.replace("muon", 0)
    data = data1.replace("electron", 1)

    #explore data, make sure it's correctly read in
    print(data.head())
    print("data[TrueLabel]:",data["particleType"])
    print("data[Prediction]:",data["Prediction"])

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(data["particleType"], data["Prediction"], classes=class_names,
                      title=model_name+' Confusion matrix, without normalization',savepath=model_name+"_cm.pdf")

    # Plot normalized confusion matrix
    plot_confusion_matrix(data["particleType"], data["Prediction"], classes=class_names, normalize=True,
                      title=model_name+' Normalized confusion matrix',savepath=model_name+"_normalized_cm.pdf")
