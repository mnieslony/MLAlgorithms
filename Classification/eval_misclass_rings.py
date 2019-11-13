import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing

#------- Read .csv files -------

var_info = pd.read_csv("predictions/RingClassification/RingCounting_AddEvInfo_Beam_FV_PMTVol_DigitThr10.csv", header = 0)
predictions = pd.read_csv("predictions/RingClassification/RandomForest_predictions_Beam_FV_PMTVol_DigitThr10.csv", header = 0)

energy = var_info.iloc[:,20]

true = predictions.iloc[:,1]
pred = predictions.iloc[:,2]

print("energy: ",energy)
print("true: ",true)
print("pred: ",pred)

#------- Select correct and wrong prediction data sets -------

energy_correct = list()
energy_correct_single = list()
energy_correct_multi = list()
energy_incorrect = list()
energy_incorrect_single = list()
energy_incorrect_multi = list()
energy_pred_correct_single = list()
energy_pred_correct_multi = list()
energy_pred_incorrect_single = list()
energy_pred_incorrect_multi = list()

for i in range(len(true)):
    #print("energy = ",energy[i])
    if true[i]==pred[i]:
        energy_correct.append(energy[i])
        if true[i]=="1-ring":
            energy_correct_single.append(energy[i])
        else:
            energy_correct_multi.append(energy[i])
        if pred[i]=="1-ring":
            energy_pred_correct_single.append(energy[i])
        else:
            energy_pred_correct_multi.append(energy[i])
    else:
        energy_incorrect.append(energy[i])
        if true[i]=="1-ring":
            energy_incorrect_single.append(energy[i])
        else:
            energy_incorrect_multi.append(energy[i])
        if pred[i]=="1-ring":
            energy_pred_incorrect_single.append(energy[i])
        else:
            energy_pred_incorrect_multi.append(energy[i])


#print(nhits_correct)
def getRatio(bin1,bin2):
    # Sanity check
    if len(bin1) != len(bin2):
        print("Cannot make ratio!")
    bins = []
    for b1,b2 in zip(bin1,bin2):
        if b1==0 and b2==0:
            bins.append(1.)
        elif b2==0:
            bins.append(0.)
        else:	
            bins.append(1-float(b1)/(float(b1)+float(b2)))
    # The ratio can of course be expanded with eg. error 
    return bins

def getRatioError(bin1,bin2):
    if len(bin1) != len(bin2):
        print("Cannot make ratio!")
    bins = []
    for b1,b2 in zip(bin1,bin2):
        if b1==0 and b2==0:
            bins.append(0.)
        elif b2 == 0:
            bins.append(0.)
        else:
            bins.append(np.sqrt((b1*b2*b2+b1*b1*b2)/(np.power(b1+b2,4))))
    #this is the expansion to calculate the errors
    return bins

#---------------------------------------------
#---histograms with ratio incorrect/correct---
#---------------------------------------------

bins_energy = np.arange(400,1400,100)
_bins, _edges = np.histogram(energy_correct,bins_energy)
_bins2, _edges2 = np.histogram(energy_incorrect,bins_energy)
bins=_bins
edges = _edges[:len(_edges)-1]
bins2 = _bins2
edges2 = _edges2[:len(_edges2)-1]
bins3 = np.sqrt(bins)
bins4 = np.sqrt(bins2)
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# Plot as colored o's, connected
ax.errorbar(edges,bins,yerr=bins3,color="blue",lw=1.5,label="correct")
ax.scatter(edges,bins,s=10,color="blue")
ax.errorbar(edges2,bins2,yerr=bins4,color="orange",lw=2,label="incorrect")
ax.scatter(edges2,bins2,s=10,color="orange")
ax.set_ylabel('#')
leg = ax.legend()
ax = fig.add_subplot(2,1,2)
rat = getRatio(bins2,bins)
error_rat = getRatioError(bins2,bins)
ax.errorbar(edges,rat,yerr=error_rat,fmt='o-',color="red")
ax.set_ylim(0.6,1)
ax.set_xlabel("energy [MeV]")
ax.set_ylabel("Accuracy")
plt.grid()
plt.savefig("RingRejection_MisclassifiedEvent_energy.pdf")
plt.close("all")

_bins, _edges = np.histogram(energy_correct_single,bins_energy)
_bins2, _edges2 = np.histogram(energy_incorrect_single,bins_energy)
bins=_bins
edges = _edges[:len(_edges)-1]
bins2 = _bins2
edges2 = _edges2[:len(_edges2)-1]
bins3 = np.sqrt(bins)
bins4 = np.sqrt(bins2)
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# Plot as colored o's, connected
ax.errorbar(edges,bins,yerr=bins3,color="blue",lw=2,label="correct (single ring)")
ax.scatter(edges,bins,s=10,color="blue")
ax.errorbar(edges2,bins2,yerr=bins4,color="orange",lw=2,label="incorrect (single ring)")
ax.scatter(edges2,bins2,s=10,color="orange")
ax.set_ylabel('#')
leg = ax.legend()
ax = fig.add_subplot(2,1,2)
rat = getRatio(bins2,bins)
error_rat = getRatioError(bins2,bins)
ax.errorbar(edges,rat,yerr=error_rat,fmt='o-',color="red")
ax.set_ylim(0.45,1)
ax.set_xlabel("energy [MeV]")
ax.set_ylabel("Accuracy")
plt.grid()
plt.savefig("RingRejection_MisclassifiedSingleRing_energy.pdf")
plt.close("all")

_bins, _edges = np.histogram(energy_correct_multi,bins_energy)
_bins2, _edges2 = np.histogram(energy_incorrect_multi,bins_energy)
bins=_bins
edges = _edges[:len(_edges)-1]
bins2 = _bins2
edges2 = _edges2[:len(_edges2)-1]
bins3 = np.sqrt(bins)
bins4 = np.sqrt(bins2)
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# Plot as colored o's, connected
ax.errorbar(edges,bins,yerr=bins3,color="blue",lw=2,label="correct (multi ring)")
ax.scatter(edges,bins,s=10,color="blue")
ax.errorbar(edges2,bins2,yerr=bins4,color="orange",lw=2,label="incorrect (multi ring)")
ax.scatter(edges2,bins2,s=10,color="orange")
ax.set_ylabel('#')
leg = ax.legend()
ax = fig.add_subplot(2,1,2)
rat = getRatio(bins2,bins)
error_rat = getRatioError(bins2,bins)
ax.errorbar(edges,rat,yerr=error_rat,fmt='o-',color="red")
ax.set_ylim(0.45,1)
ax.set_xlabel("energy [MeV]")
ax.set_ylabel("Accuracy")
plt.grid()
plt.savefig("RingRejection_MisclassifiedMultiRingEvent_energy.pdf")
plt.close("all")

_bins, _edges = np.histogram(energy_pred_correct_single,bins_energy)
_bins2, _edges2 = np.histogram(energy_pred_incorrect_single,bins_energy)
bins=_bins
edges = _edges[:len(_edges)-1]
bins2 = _bins2
edges2 = _edges2[:len(_edges2)-1]
bins3 = np.sqrt(bins)
bins4 = np.sqrt(bins2)
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# Plot as colored o's, connected
ax.errorbar(edges,bins,yerr=bins3,color="blue",lw=2,label="correct (single ring)")
ax.scatter(edges,bins,s=10,color="blue")
ax.errorbar(edges2,bins2,yerr=bins4,color="orange",lw=2,label="incorrect (single ring)")
ax.scatter(edges2,bins2,s=10,color="orange")
ax.set_ylabel('#')
leg = ax.legend()
ax = fig.add_subplot(2,1,2)
rat = getRatio(bins2,bins)
error_rat = getRatioError(bins2,bins)
ax.errorbar(edges,rat,yerr=error_rat,fmt='o-',color="red")
ax.set_ylim(0.45,1)
ax.set_xlabel("energy [MeV]")
ax.set_ylabel("Precision")
plt.grid()
plt.savefig("RingRejection_PredictedSingleRing_energy.pdf")
plt.close("all")

_bins, _edges = np.histogram(energy_pred_correct_multi,bins_energy)
_bins2, _edges2 = np.histogram(energy_pred_incorrect_multi,bins_energy)
bins=_bins
edges = _edges[:len(_edges)-1]
bins2 = _bins2
edges2 = _edges2[:len(_edges2)-1]
bins3 = np.sqrt(bins)
bins4 = np.sqrt(bins2)
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# Plot as colored o's, connected
ax.errorbar(edges,bins,yerr=bins3,color="blue",lw=2,label="correct (multi ring)")
ax.scatter(edges,bins,s=10,color="blue")
ax.errorbar(edges2,bins2,yerr=bins4,color="orange",lw=2,label="incorrect (multi ring)")
ax.scatter(edges2,bins2,s=10,color="orange")
ax.set_ylabel('#')
leg = ax.legend()
ax = fig.add_subplot(2,1,2)
rat = getRatio(bins2,bins)
error_rat = getRatioError(bins2,bins)
ax.errorbar(edges,rat,yerr=error_rat,fmt='o-',color="red")
ax.set_ylim(0.45,1)
ax.set_xlabel("energy [MeV]")
ax.set_ylabel("Precision")
plt.grid()
plt.savefig("RingRejection_PredictedMultiRingEvent_energy.pdf")
plt.close("all")
