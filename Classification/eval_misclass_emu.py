import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing

#------- Read .csv files -------

var_info = pd.read_csv("PID_AddEvInfo_Beamlike_FV_PMTVol_DigitThr10.csv", header = 0)
predictions = pd.read_csv("PID_MLP_predictions_Beamlike_FV_PMTVol_DigitThr10.csv", header = 0)

true = predictions.iloc[:,1]
pred = predictions.iloc[:,2]

print("nhits: ",var_info['pmtHits'])
print("charge: ",var_info['pmtTotalQ'])
print("mrdclusters: ",var_info['mrdClusters'])
print("energy: ",var_info['energy'])
print("true: ",predictions['particleType'])
print("pred: ",predictions['Prediction'])

correct = list()
incorrect = list()
correct_e = list()
correct_mu = list()
incorrect_e = list()
incorrect_mu = list()

def getRatio(bin1,bin2):
    if len(bin1) != len(bin2):
        print("Cannot make ratio!")
    bins = []
    for b1,b2 in zip(bin1,bin2):
        if b1==0 and b2==0:
            bins.append(0.)
        elif b2==0:
            bins.append(0.)
        else:	
            bins.append(1-float(b1)/(float(b1)+float(b2)))
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
    return bins

def create_lists(varname):
    #------- Select correct and wrong prediction data sets -------

    variable = var_info[varname]

    for i in range(len(true)):
        if true[i]==pred[i]:
            correct.append(variable[i])
            if true[i]=="muon":
                correct_mu.append(variable[i])
            else:
                correct_e.append(variable[i])
        else:
            incorrect.append(variable[i])
            if true[i]=="muon":
                incorrect_mu.append(variable[i])
            else:
                incorrect_e.append(variable[i])

def plot_ratios(lower,upper,bin_width,varname,plot_type):
    
    bins_var = np.arange(lower,upper,bin_width)
    if plot_type is 'electron':
        _bins, _edges = np.histogram(correct_e,bins_var)
        _bins2, _edges2 = np.histogram(incorrect_e,bins_var)
    elif plot_type is 'muon':
        _bins, _edges = np.histogram(correct_mu,bins_var)
        _bins2, _edges2 = np.histogram(incorrect_mu,bins_var)
    else:
        #default: overall plot
        _bins, _edges = np.histogram(correct,bins_var)
        _bins2, _edges2 = np.histogram(incorrect,bins_var)
    bins=_bins
    edges = _edges[:len(_edges)-1]
    bins2 = _bins2
    edges2 = _edges2[:len(_edges2)-1]
    bins3 = np.sqrt(bins)
    bins4 = np.sqrt(bins2)
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    label_correct = "correct"
    label_incorrect = "incorrect"
    if plot_type is 'electron':
        label_correct = "correct (e)"
        label_incorrect = "incorrect (e)"
    elif plot_type is 'muon':
        label_correct = "correct (mu)"
        label_incorrect = "incorrect (mu)"
    ax.errorbar(edges,bins,yerr=bins3,color="blue",lw=2,label=label_correct)
    ax.scatter(edges,bins,s=10,color="blue")
    ax.errorbar(edges2,bins2,yerr=bins4,color="orange",lw=2,label=label_incorrect)
    ax.scatter(edges2,bins2,s=10,color="orange")
    ax.set_ylabel('#')
    leg = ax.legend()
    if plot_type is 'electron':
        ax.set_title("Misclassified electrons")
    elif plot_type is 'muon':
        ax.set_title('Misclassified muons')
    else:
        ax.set_title('Misclassified events')
    ax = fig.add_subplot(2,1,2)
    rat = getRatio(bins2,bins)
    error_rat = getRatioError(bins2,bins)
    ax.errorbar(edges,rat,yerr=error_rat,fmt='o-',color='red')
    ax.set_ylim(min(rat)-0.05,max(rat)+0.05)
    ax.set_xlabel(varname)
    ax.set_ylabel("Accuracy")
                     
    plt.grid()
    if plot_type is 'electron':
        plt.savefig("PID_MisclassifiedE_Ratio_"+varname+".pdf")
    elif plot_type is 'muon':
        plt.savefig("PID_MisclassifiedMu_Ratio_"+varname+".pdf")
    else:
        plt.savefig("PID_Misclassified_Ratio_"+varname+".pdf")
    plt.close("all")

def plot_histograms(lower,upper,bin_width,varname,plot_type):
    bins_var=np.arange(lower,upper,bin_width)
    if plot_type is 'electron':
        plt.hist(correct_e,bins_var,label='correct (e)')
        plt.hist(incorrect_e,bins_var,label='incorrect (e)')
    elif plot_type is 'muon':
        plt.hist(correct_mu,bins_var,label='correct (mu)')
        plt.hist(incorrect_mu,bins_var,label='incorrect (mu)')
    else:
        plt.hist(correct,bins_var,label='correct')
        plt.hist(incorrect,bins_var,label='incorrect')
    plt.xlabel(varname)
    plt.ylabel('#')
    if plot_type is 'electron':
        plt.title("Misclassified electrons")
    elif plot_type is 'muon':
        plt.title('Misclassified muons')
    else:
        plt.title('Misclassified events')
    plt.legend(loc='upper left')
    if plot_type is 'electron':
        plt.savefig("PID_MisclassifiedE_Hist_"+varname+".pdf")
    elif plot_type is 'muon':
        plt.savefig("PID_MisclassifiedMu_Hist_"+varname+".pdf")
    else:
        plt.savefig("PID_Misclassified_Hist_"+varname+".pdf")
    plt.close("all")

    
var_types=['pmtHits','pmtTotalQ','mrdClusters','energy']
lower_bins = [10,0,0,100]
upper_bins = [140,6000,5,2500]
bin_widths = [14,300,1,280]
plot_types = ['overall','electron','muon']

for i in range(len(var_types)):
    create_lists(var_types[i])
    for plottype in plot_types:
        plot_histograms(lower_bins[i],upper_bins[i],bin_widths[i],var_types[i],plottype)
        plot_ratios(lower_bins[i],upper_bins[i],bin_widths[i],var_types[i],plottype)


