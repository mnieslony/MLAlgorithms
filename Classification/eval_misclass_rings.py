import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing

#------- Read .csv files -------

var_info = pd.read_csv("predictions/RingClassification/RingCounting_AddEvInfo_Beam_FV_PMTVol_DigitThr10.csv", header = 0)
predictions = pd.read_csv("predictions/RingClassification/RingCounting_RandomForest_predictions_FV_PMTVol_DigitThr10.csv", header = 0)

true = predictions.iloc[:,1]
pred = predictions.iloc[:,2]

print("true: ",true)
print("pred: ",pred)
print("energy: ",var_info['energy'])

#------- Select correct and wrong prediction data sets -------

correct = list()
correct_single = list()
correct_multi = list()
pred_correct = list()
pred_correct_single = list()
pred_correct_multi = list()
incorrect = list()
incorrect_single = list()
incorrect_multi = list()
pred_incorrect = list()
pred_incorrect_single = list()
pred_incorrect_multi = list()

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
            if true[i]=="1-ring":
                correct_single.append(variable[i])
            else:
                correct_multi.append(variable[i])
            if pred[i]=="1-ring":
                pred_correct_single.append(variable[i])
            else:
                pred_correct_multi.append(variable[i])
        else:
            incorrect.append(variable[i])
            if true[i]=="1-ring":
                incorrect_single.append(variable[i])
            else:
                incorrect_multi.append(variable[i])
            if pred[i]=="1-ring":
                pred_incorrect_single.append(variable[i])
            else:
                pred_incorrect_multi.append(variable[i])


def plot_ratios(lower,upper,bin_width,varname,plot_type):

    print("Executing plot_ratios: plot_type = ",plot_type)
    
    bins_var = np.arange(lower,upper,bin_width)
    if plot_type == '1-ring':
        _bins, _edges = np.histogram(correct_single,bins_var)
        _bins2, _edges2 = np.histogram(incorrect_single,bins_var)
    elif plot_type == 'multi-ring':
        _bins, _edges = np.histogram(correct_multi,bins_var)
        _bins2, _edges2 = np.histogram(incorrect_multi,bins_var)
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
    if plot_type == '1-ring':
        label_correct = "correct (1-ring)"
        label_incorrect = "incorrect (1-ring)"
    elif plot_type == 'multi-ring':
        label_correct = "correct (multi-ring)"
        label_incorrect = "incorrect (multi-ring)"
    ax.errorbar(edges,bins,yerr=bins3,color="blue",lw=2,label=label_correct)
    ax.scatter(edges,bins,s=10,color="blue")
    ax.errorbar(edges2,bins2,yerr=bins4,color="orange",lw=2,label=label_incorrect)
    ax.scatter(edges2,bins2,s=10,color="orange")
    ax.set_ylabel('#')
    leg = ax.legend()
    if plot_type == '1-ring':
        ax.set_title("Misclassified single rings")
    elif plot_type == 'multi-ring':
        ax.set_title('Misclassified multi rings')
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
    if plot_type == '1-ring':
        plt.savefig("RingClassification_MisclassifiedSingle_Ratio_"+varname+".pdf")
    elif plot_type == 'multi-ring':
        plt.savefig("RingClassification_MisclassifiedMulti_Ratio_"+varname+".pdf")
    else:
        plt.savefig("RingClassification_Misclassified_Ratio_"+varname+".pdf")
    plt.close("all")

def plot_precision(lower,upper,bin_width,varname,plot_type):

    print("Executing plot_precision: plot_type = ",plot_type)

    bins_var = np.arange(lower,upper,bin_width)
    if plot_type == '1-ring':
        _bins, _edges = np.histogram(pred_correct_single,bins_var)
        _bins2, _edges2 = np.histogram(pred_incorrect_multi,bins_var)
    else:
        _bins, _edges = np.histogram(pred_correct_multi,bins_var)
        _bins2, _edges2 = np.histogram(pred_incorrect_single,bins_var)
    bins=_bins
    edges = _edges[:len(_edges)-1]
    bins2 = _bins2
    edges2 = _edges2[:len(_edges2)-1]
    bins3 = np.sqrt(bins)
    bins4 = np.sqrt(bins2)
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    if plot_type == '1-ring':
        label_correct = "correct (1-ring)"
        label_incorrect = "incorrect (1-ring)"
    else:
        label_correct = "correct (multi-ring)"
        label_incorrect = "incorrect (multi-ring)"
    ax.errorbar(edges,bins,yerr=bins3,color="blue",lw=2,label=label_correct)
    ax.scatter(edges,bins,s=10,color="blue")
    ax.errorbar(edges2,bins2,yerr=bins4,color="orange",lw=2,label=label_incorrect)
    ax.scatter(edges2,bins2,s=10,color="orange")
    ax.set_ylabel('#')
    leg = ax.legend()
    if plot_type == '1-ring':
        ax.set_title("Predicted single rings")
    else:
        ax.set_title('Predicted multi rings')
    ax = fig.add_subplot(2,1,2)
    rat = getRatio(bins2,bins)
    error_rat = getRatioError(bins2,bins)
    ax.errorbar(edges,rat,yerr=error_rat,fmt='o-',color='red')
    ax.set_ylim(min(rat)-0.05,max(rat)+0.05)
    ax.set_xlabel(varname)
    ax.set_ylabel("Precision")
                     
    plt.grid()
    if plot_type == '1-ring':
        plt.savefig("RingClassification_MisclassifiedSingle_Precision_"+varname+".pdf")
    else:
        plt.savefig("RingClassification_MisclassifiedMulti_Precision_"+varname+".pdf")
    plt.close("all")


def plot_histograms(lower,upper,bin_width,varname,plot_type):
    print("Executing plot_histograms: plot_type = ",plot_type)
    bins_var=np.arange(lower,upper,bin_width)
    if plot_type == '1-ring':
        plt.hist(correct_single,bins_var,label='correct (1-ring)')
        plt.hist(incorrect_single,bins_var,label='incorrect (1-ring)')
    elif plot_type == 'multi-ring':
        plt.hist(correct_multi,bins_var,label='correct (multi ring)')
        plt.hist(incorrect_multi,bins_var,label='incorrect (multi ring)')
    else:
        plt.hist(correct,bins_var,label='correct')
        plt.hist(incorrect,bins_var,label='incorrect')
    plt.xlabel(varname)
    plt.ylabel('#')
    if plot_type == '1-ring':
        plt.title("Misclassified 1-rings")
    elif plot_type == 'multi-ring':
        plt.title('Misclassified multi rings')
    else:
        plt.title('Misclassified events')
    plt.legend(loc='upper left')
    if plot_type == '1-ring':
        plt.savefig("RingClassification_MisclassifiedSingle_Hist_"+varname+".pdf")
    elif plot_type == 'multi-ring':
        plt.savefig("RingClassification_MisclassifiedMulti_Hist_"+varname+".pdf")
    else:
        plt.savefig("RingClassification_Misclassified_Hist_"+varname+".pdf")
    plt.close("all")

var_types=['pmtHits','energy']
lower_bins = [10,100]
upper_bins = [80,2000]
bin_widths = [5,100]
plot_types = ['overall','1-ring','multi-ring']

for i in range(len(var_types)):
    create_lists(var_types[i])
    for plottype in plot_types:
        plot_histograms(lower_bins[i],upper_bins[i],bin_widths[i],var_types[i],plottype)
        plot_ratios(lower_bins[i],upper_bins[i],bin_widths[i],var_types[i],plottype)
        if plottype == '1-ring' or plottype == 'multi-ring':
            plot_precision(lower_bins[i],upper_bins[i],bin_widths[i],var_types[i],plottype)

