import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing

#------- Read .csv files -------

var_info = pd.read_csv("AddEvInfo_BeamlikeSample_FV_MRDCluster_NoMC.csv", header = None)
predictions = pd.read_csv("DecisionTree_predictions_BeamlikeSample_FV_MRDCluster_NoMC.csv", header = None)

nhits = var_info.iloc[:,1]
charge = var_info.iloc[:,2]
mrdclusters = var_info.iloc[:,29]
energy = var_info.iloc[:,30]

true = predictions.iloc[:,1]
pred = predictions.iloc[:,2]

print("nhits: ",nhits)
print("charge: ",charge)
print("mrdclusters: ",mrdclusters)
print("energy: ",energy)
print("true: ",true)
print("pred: ",pred)

#------- Select correct and wrong prediction data sets -------

nhits_correct = list()
charge_correct = list()
energy_correct = list()
energy_correct_e = list()
energy_correct_mu = list()
mrdclusters_correct = list()
mrdclusters_correct_e = list()
mrdclusters_correct_mu = list()
nhits_incorrect = list()
charge_incorrect = list()
energy_incorrect = list()
energy_incorrect_e = list()
energy_incorrect_mu = list()
mrdclusters_incorrect = list()
mrdclusters_incorrect_e = list()
mrdclusters_incorrect_mu = list()

for i in range(len(true)):
    if true[i]==pred[i]:
        nhits_correct.append(nhits[i])
        charge_correct.append(charge[i])
        energy_correct.append(energy[i])
        mrdclusters_correct.append(mrdclusters[i])
        if true[i]=="muon":
            energy_correct_mu.append(energy[i])
            mrdclusters_correct_mu.append(mrdclusters[i])
        else:
            energy_correct_e.append(energy[i])
            mrdclusters_correct_e.append(mrdclusters[i])
    else:
        nhits_incorrect.append(nhits[i])
        charge_incorrect.append(charge[i])
        energy_incorrect.append(energy[i])
        mrdclusters_incorrect.append(mrdclusters[i])
        if true[i]=="muon":
            energy_incorrect_mu.append(energy[i])
            mrdclusters_incorrect_mu.append(mrdclusters[i])
        else:
            energy_incorrect_e.append(energy[i])
            mrdclusters_incorrect_e.append(mrdclusters[i])

error_energy_correct = np.sqrt(energy_correct)
error_energy_incorrect = np.sqrt(energy_incorrect)
error_energy_correct_e = np.sqrt(energy_correct_e)
error_energy_incorrect_e = np.sqrt(energy_incorrect_e)
error_energy_correct_mu = np.sqrt(energy_correct_mu)
error_energy_incorrect_mu = np.sqrt(energy_incorrect_mu)
error_mrdclusters_correct = np.sqrt(mrdclusters_correct)
error_mrdclusters_incorrect = np.sqrt(mrdclusters_incorrect)
error_mrdclusters_correct_e = np.sqrt(mrdclusters_correct_e)
error_mrdclusters_incorrect_e = np.sqrt(mrdclusters_incorrect_e)
error_mrdclusters_correct_mu = np.sqrt(mrdclusters_correct_mu)
error_mrdclusters_incorrect_mu = np.sqrt(mrdclusters_incorrect_mu)
    
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

bins_nhits = np.arange(10,140,14)
_bins, _edges = np.histogram(nhits_correct,bins_nhits)
_bins2, _edges2 = np.histogram(nhits_incorrect,bins_nhits)
bins=_bins
edges = _edges[:len(_edges)-1]
bins2 = _bins2
edges2 = _edges2[:len(_edges2)-1]
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# Plot as colored o's, connected
ax.plot(edges,bins,'o-',color="blue",lw=2,label="correct")
ax.plot(edges2,bins2,'o-',color="orange",lw=2,label="incorrect")
ax.set_ylabel('#')
leg = ax.legend()
ax = fig.add_subplot(2,1,2)
rat = getRatio(bins2,bins)
ax.plot(edges,rat)
ax.set_ylim(0,1)
ax.set_xlabel("n hit PMTs")
ax.set_ylabel("Accuracy")
plt.savefig("MisclassifiedEvents_hitPMTs.pdf")
plt.close("all")


bins_charge = np.arange(0,12500,625)
_bins, _edges = np.histogram(charge_correct,bins_charge)
_bins2, _edges2 = np.histogram(charge_incorrect,bins_charge)
bins=_bins
edges = _edges[:len(_edges)-1]
bins2 = _bins2
edges2 = _edges2[:len(_edges2)-1]
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# Plot as colored o's, connected
ax.plot(edges,bins,'o-',color="blue",lw=2,label="correct")
ax.plot(edges2,bins2,'o-',color="orange",lw=2,label="incorrect")
ax.set_ylabel('#')
leg = ax.legend()
ax = fig.add_subplot(2,1,2)
rat = getRatio(bins2,bins)
ax.plot(edges,rat)
ax.set_ylim(0,1)
ax.set_xlabel("charge [p.e.]")
ax.set_ylabel("Accuracy")
plt.savefig("MisclassifiedEvent_charge.pdf")
plt.close("all")

bins_energy = np.arange(100,2500,280)
_bins, _edges = np.histogram(energy_correct,bins_energy)
_bins2, _edges2 = np.histogram(energy_incorrect,bins_energy)
_bins3, _edges3 = np.histogram(error_energy_correct,bins_energy)
_bins4, _edges4 = np.histogram(error_energy_incorrect,bins_energy)
bins=_bins
edges = _edges[:len(_edges)-1]
bins2 = _bins2
edges2 = _edges2[:len(_edges2)-1]
edges3 = np.sqrt(edges)
edges4 = np.sqrt(edges2)
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# Plot as colored o's, connected
ax.errorbar(edges,bins,yerr=edges3,fmt='o-',color="blue",lw=2,label="correct")
ax.errorbar(edges2,bins2,yerr=edges4,fmt='o-',color="orange",lw=2,label="incorrect")
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
plt.savefig("MisclassifiedEvent_energy.pdf")
plt.close("all")

_bins, _edges = np.histogram(energy_correct_e,bins_energy)
_bins2, _edges2 = np.histogram(energy_incorrect_e,bins_energy)
_bins3, _edges3 = np.histogram(error_energy_correct_e,bins_energy)
_bins4, _edges4 = np.histogram(error_energy_incorrect_e,bins_energy)
bins=_bins
edges = _edges[:len(_edges)-1]
bins2 = _bins2
edges2 = _edges2[:len(_edges2)-1]
edges3 = np.sqrt(edges)
edges4 = np.sqrt(edges2)
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# Plot as colored o's, connected
ax.errorbar(edges,bins,yerr=edges3,fmt='o-',color="blue",lw=2,label="correct (e-)")
ax.errorbar(edges2,bins2,yerr=edges4,fmt='o-',color="orange",lw=2,label="incorrect (e-)")
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
plt.savefig("MisclassifiedElectronEvent_energy.pdf")
plt.close("all")

_bins, _edges = np.histogram(energy_correct_mu,bins_energy)
_bins2, _edges2 = np.histogram(energy_incorrect_mu,bins_energy)
_bins3, _edges3 = np.histogram(error_energy_correct_mu,bins_energy)
_bins4, _edges4 = np.histogram(error_energy_incorrect_mu,bins_energy)
bins=_bins
edges = _edges[:len(_edges)-1]
bins2 = _bins2
edges2 = _edges2[:len(_edges2)-1]
edges3 = np.sqrt(edges)
edges4 = np.sqrt(edges2)
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# Plot as colored o's, connected
ax.errorbar(edges,bins,yerr=edges3,fmt='o-',color="blue",lw=2,label="correct (mu-)")
ax.errorbar(edges2,bins2,yerr=edges4,fmt='o-',color="orange",lw=2,label="incorrect (mu-)")
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
plt.savefig("MisclassifiedMuonEvent_energy.pdf")
plt.close("all")

bins_mrdclusters = np.arange(0,5,1)
_bins, _edges = np.histogram(mrdclusters_correct,bins_mrdclusters)
_bins2, _edges2 = np.histogram(mrdclusters_incorrect,bins_mrdclusters)
_bins3, _edges3 = np.histogram(error_mrdclusters_correct,bins_mrdclusters)
_bins4, _edges4 = np.histogram(error_mrdclusters_incorrect,bins_mrdclusters)
bins=_bins
edges = _edges[:len(_edges)-1]
bins2 = _bins2
edges2 = _edges2[:len(_edges2)-1]
edges3 = np.sqrt(edges)
edges4 = np.sqrt(edges2)
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# Plot as colored o's, connected
ax.errorbar(edges,bins,yerr=edges3,fmt='o-',color="blue",lw=2,label="correct")
ax.errorbar(edges2,bins2,yerr=edges4,fmt='o-',color="orange",lw=2,label="incorrect")
ax.set_ylabel('#')
leg = ax.legend()
ax = fig.add_subplot(2,1,2)
rat = getRatio(bins2,bins)
error_rat = getRatioError(bins2,bins)
ax.errorbar(edges,rat,yerr=error_rat,fmt='o-',color="red")
ax.set_ylim(0.6,1.1)
ax.set_xlabel("# MRD clusters")
ax.set_ylabel("Accuracy")
plt.grid()
plt.savefig("MisclassifiedEvent_mrdclusters.pdf")
plt.close("all")

_bins, _edges = np.histogram(mrdclusters_correct_e,bins_mrdclusters)
_bins2, _edges2 = np.histogram(mrdclusters_incorrect_e,bins_mrdclusters)
_bins3, _edges3 = np.histogram(error_mrdclusters_correct_e,bins_mrdclusters)
_bins4, _edges4 = np.histogram(error_mrdclusters_incorrect_e,bins_mrdclusters)
bins=_bins
edges = _edges[:len(_edges)-1]
bins2 = _bins2
edges2 = _edges2[:len(_edges2)-1]
edges3 = np.sqrt(edges)
edges4 = np.sqrt(edges2)
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# Plot as colored o's, connected
ax.errorbar(edges,bins,yerr=edges3,fmt='o-',color="blue",lw=2,label="correct (e-)")
ax.errorbar(edges2,bins2,yerr=edges4,fmt='o-',color="orange",lw=2,label="incorrect (e-)")
ax.set_ylabel('#')
leg = ax.legend()
ax = fig.add_subplot(2,1,2)
rat = getRatio(bins2,bins)
error_rat = getRatioError(bins2,bins)
ax.errorbar(edges,rat,yerr=error_rat,fmt='o-',color="red")
ax.set_ylim(0.0,1.1)
ax.set_xlabel("# MRD clusters")
ax.set_ylabel("Accuracy")
plt.grid()
plt.savefig("MisclassifiedElectronEvent_mrdclusters.pdf")
plt.close("all")

_bins, _edges = np.histogram(mrdclusters_correct_mu,bins_mrdclusters)
_bins2, _edges2 = np.histogram(mrdclusters_incorrect_mu,bins_mrdclusters)
_bins3, _edges3 = np.histogram(error_mrdclusters_correct_mu,bins_mrdclusters)
_bins4, _edges4 = np.histogram(error_mrdclusters_incorrect_mu,bins_mrdclusters)
bins=_bins
edges = _edges[:len(_edges)-1]
bins2 = _bins2
edges2 = _edges2[:len(_edges2)-1]
edges3 = np.sqrt(edges)
edges4 = np.sqrt(edges2)
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# Plot as colored o's, connected
ax.errorbar(edges,bins,yerr=edges3,fmt='o-',color="blue",lw=2,label="correct (mu-)")
ax.errorbar(edges2,bins2,yerr=edges4,fmt='o-',color="orange",lw=2,label="incorrect (mu-)")
ax.set_ylabel('#')
leg = ax.legend()
ax = fig.add_subplot(2,1,2)
rat = getRatio(bins2,bins)
error_rat = getRatioError(bins2,bins)
ax.errorbar(edges,rat,yerr=error_rat,fmt='o-',color="red")
ax.set_ylim(0.6,1.1)
ax.set_xlabel("# MRD clusters")
ax.set_ylabel("Accuracy")
plt.grid()
plt.savefig("MisclassifiedMuonEvent_mrdclusters.pdf")
plt.close("all")

#---------------------------------------------
#-histograms without ratio incorrect/correct--
#---------------------------------------------

plt.hist(nhits_correct,bins_nhits,label='correct')
plt.hist(nhits_incorrect,bins_nhits,label='incorrect')
plt.xlabel('hit PMTs')
plt.ylabel('#')
plt.legend(loc='upper left')
#plt.show()
plt.savefig("Misclassified_hitPMTs.pdf")
plt.close("all")

plt.hist(charge_correct,bins_charge, label='correct')
plt.hist(charge_incorrect,bins_charge, label='incorrect')
plt.xlabel('charge [p.e.]')
plt.ylabel('#')
plt.legend(loc='upper right')
#plt.show()
plt.savefig("Misclassified_charge.pdf")
plt.close("all")

plt.hist(energy_correct,bins_energy,label='correct')
plt.hist(energy_incorrect,bins_energy,label='incorrect')
plt.xlabel('energy [MeV]')
plt.ylabel('#')
plt.legend(loc='upper left')
#plt.show()
plt.savefig("Misclassified_energy.pdf")
plt.close("all")

