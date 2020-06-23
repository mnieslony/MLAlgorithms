import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing

import argparse

#------- Parse user arguments ----

parser = argparse.ArgumentParser(description='Ring Classification Energy Dependence - Overview')
parser.add_argument("--model_name",default="MLP", help = "The name of the model to evaluate. Options: RandomForest, XGBoost, SVM, SGD, MLP, GradientBoosting, All")
parser.add_argument("--variable_config", default="Old", help = "Classification variable configuration name")
parser.add_argument("--plot_variables", default="VariableConfig_Plot_Old.txt", help = "File containing the list of dependency variables")
parser.add_argument("--dataset_name",default="beam", help = "Keyword describing dataset name (used to label output files)")

args = parser.parse_args()
model_name = args.model_name
variable_config = args.variable_config
plot_variables = args.plot_variables
dataset_name = args.dataset_name

#------- Read .csv files -------

filename_additional = "additional_event_info.nosync/RingClassification/RingClassification_AddEvInfo_"+dataset_name+"_"+variable_config+".csv"
filename_predictions = "predictions/RingClassification/RingClassification_"+model_name+"_predictions_"+dataset_name+"_"+variable_config+".csv"

var_info = pd.read_csv(filename_additional, header = 0)
predictions = pd.read_csv(filename_predictions, header = 0)

# -------- Retrieve information from predictions-file ----

true = predictions.iloc[:,1]
pred = predictions.iloc[:,2]

# ----- Retrieve information from additional-information-file -----

list_plot_variables=[]
list_lowerbin=[]
list_upperbin=[]
list_binwidth=[]

create_plots = True

plot_variables_path = "variable_config/"+plot_variables
with open(plot_variables_path) as f:
    for line in f:
        currentline = line.split(",")
        print('length of currentline: ',len(currentline))
        if len(currentline)!=4:
            print('Error in plot_variables configuration file '+plot_variables+'! Rows have length of '+len(currentline)+' instead of 4!')
            create_plots = False
        if create_plots:
            list_plot_variables.append(currentline[0])
            list_lowerbin.append(int(currentline[1]))
            list_upperbin.append(int(currentline[2]))
            list_binwidth.append(int(currentline[3]))

print('Variables to be used in the plots: ',list_plot_variables)

for variable in list_plot_variables:

    print('Preview data for variable ',variable,': ',var_info[variable].head())


#print("Preview list of true classes: ",predictions['multiplerings'].head())
#print("Preview list of predicted classes: ",predictions['multiplerings'].head())
print("Preview list of true classes: ",predictions['MCMultiRing'].head())
print("Preview list of predicted classes: ",predictions['MCMultiRing'].head())

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

# ------------------------------------------------------------------
# ------------ getRatio: Get ratio of two histograms ---------------
# ------------------------------------------------------------------

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

# ------------------------------------------------------------------
# ------ getRatioError: Get error of ratio of two histograms -------
# ------------------------------------------------------------------  

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

# -----------------------------------------------------------------------------------
# ------ create_lists: Create lists of (in)correctly predicted particle types -------
# -----------------------------------------------------------------------------------

def create_lists(varname):
    #------- Select correct and wrong prediction data sets -------

    variable = var_info[varname]

    for i in range(len(true)):
        if true[i]==pred[i]:
            correct.append(variable[i])
            if true[i]=="1-ring":
                correct_single.append(variable[i])
                pred_correct_single.append(variable[i])
            else:
                correct_multi.append(variable[i])
                pred_correct_multi.append(variable[i])
        else:
            incorrect.append(variable[i])
            if true[i]=="1-ring":
                incorrect_single.append(variable[i])
                pred_incorrect_multi.append(variable[i])
            else:
                incorrect_multi.append(variable[i])
                pred_incorrect_single.append(variable[i])

# -----------------------------------------------------------------------------------
# ------ create_lists: Special list creation for Pi+/Pi-/Pi0/K+/K- energies ---------
# -----------------------------------------------------------------------------------

def create_lists_pip():

    variable=var_info["Pip"]
    variableE = var_info["Pim"]

    for i in range(len(true)):
        if true[i]=="multi-ring":
            if variable[i]==1:
                if pred[i]=="multi-ring":
                    correct.append(variableE[i])
                else:
                    incorrect.append(variableE[i])

def create_lists_pim():

    variable1=var_info["Pip"]
    variable2=var_info["Pim"]
    variable3=var_info["Pi0"]

    for i in range(len(true)):
        if true[i]=="multi-ring":
            if variable1[i]==0:
                if variable2[i]==1:
                    if pred[i]=="multi-ring":
                        correct.append(variable3[i])
                    else:
                        incorrect.append(variable3[i])

def create_lists_pi0():

    variable1=var_info["Pip"]
    variable2=var_info["Pim"]
    variable3=var_info["Pi0"]
    variable4=var_info["Kp"]

    for i in range(len(true)):
        if true[i]=="multi-ring":
            if variable1[i]==0:
                if variable2[i]==0:
                    if variable3[i]==1:
                        if pred[i]=="multi-ring":
                            correct.append(variable4[i])
                        else:
                            incorrect.append(variable4[i])

def create_lists_kp():

    variable1=var_info["Pip"]
    variable2=var_info["Pim"]
    variable3=var_info["Pi0"]
    variable4=var_info["Kp"]
    variable5=var_info["Km"]

    for i in range(len(true)):
        if true[i]=="multi-ring":
            if variable1[i]==0:
                if variable2[i]==0:
                    if variable3[i]==0:
                        if variable4[i]==1:
                            if pred[i]=="multi-ring":
                                correct.append(variable5[i])
                            else:
                                incorrect.append(variable5[i])

def create_lists_km():

    variable1=var_info["Pip"]
    variable2=var_info["Pim"]
    variable3=var_info["Pi0"]
    variable4=var_info["Kp"]
    variable5=var_info["Km"]
    variable6=var_info["Column1"]

    for i in range(len(true)):
        if true[i]=="multi-ring":
            if variable1[i]==0:
                if variable2[i]==0:
                    if variable3[i]==0:
                        if variable4[i]==0:
                            if variable5[i]==1:
                                if pred[i]=="multi-ring":
                                    correct.append(variable6[i])
                                else:
                                    incorrect.append(variable6[i])

# ---------------------------------------------------------------------------------
# ------ plot_ratios: Plot ratios of correctly classified events (variable) -------
# ---------------------------------------------------------------------------------

def plot_ratios(lower,upper,bin_width,varname,plot_type):

    print("Executing plot_ratios: plot_type = ",plot_type)

    acc_file = open("plots/RingClassification/EnergyDependence/Accuracy_RingClassification_"+varname+"_"+plot_type+"_"+model_name+"_"+dataset_name+"_"+variable_config+".csv","w")
    acc_file.write("plot_type,lower_bin,n_total,n_incorrect,accuracy,error_accuracy\n")
    
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

    ax.errorbar(edges,bins,yerr=bins3,fmt='',color="blue",lw=1,label=label_correct)
    ax.scatter(edges,bins,s=5,color="blue")
    ax.errorbar(edges2,bins2,yerr=bins4,fmt='',color="orange",lw=1,label=label_incorrect)
    ax.scatter(edges2,bins2,s=5,color="orange")
    ax.set_ylabel('#')
    leg = ax.legend()

    if plot_type == '1-ring':
        ax.set_title("Misclassified single rings ("+dataset_name+" dataset)")
    elif plot_type == 'multi-ring':
        ax.set_title('Misclassified multi rings ('+dataset_name+' dataset)')
    else:
        ax.set_title('Misclassified events ('+dataset_name+' dataset)')

    ax = fig.add_subplot(2,1,2)
    rat = getRatio(bins2,bins)
    error_rat = getRatioError(bins2,bins)
    plt.grid()
    ax.set_axisbelow(True)
    ax.errorbar(edges,rat,yerr=error_rat,fmt='none',lw=1,color='red')
    ax.scatter(edges,rat,s=5,color='red')
    #ax.set_ylim(min(rat)-0.05,max(rat)+0.05)
    ax.set_ylim(-0.05,1.1)
    ax.set_xlabel(varname)
    ax.set_ylabel("Accuracy")
                     
    if plot_type == '1-ring':
        plt.savefig("plots/RingClassification/EnergyDependence/RingClassification_Classification_Ratio_"+dataset_name+"_"+variable_config+"_"+varname+"_single.pdf")
    elif plot_type == 'multi-ring':
        plt.savefig("plots/RingClassification/EnergyDependence/RingClassification_Classification_Ratio_"+dataset_name+"_"+variable_config+"_"+varname+"_multi.pdf")
    else:
        plt.savefig("plots/RingClassification/EnergyDependence/RingClassification_Classification_Ratio_"+dataset_name+"_"+variable_config+"_"+varname+"_overall.pdf")
    plt.close("all")

    for i in range(len(edges)):
        acc_file.write(plot_type+","+str(edges[i])+","+str(bins[i])+","+str(bins2[i])+","+str(rat[i])+","+str(error_rat[i])+"\n")

# ------------------------------------------------------------------------------------------
# ------ plot_precision: Plot histograms for precision of classification (variable) --------
# ------------------------------------------------------------------------------------------

def plot_precision(lower,upper,bin_width,varname,plot_type):

    print("Executing plot_precision: plot_type = ",plot_type)

    prec_file = open("plots/RingClassification/EnergyDependence/Precision_RingClassification_"+varname+"_"+plot_type+"_"+model_name+"_"+dataset_name+"_"+variable_config+".csv","w")
    prec_file.write("plot_type,lower_bin,n_total,n_incorrect,accuracy,error_accuracy\n")

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

    ax.errorbar(edges,bins,yerr=bins3,color="blue",fmt='',lw=1,label=label_correct)
    ax.scatter(edges,bins,s=5,color="blue")
    ax.errorbar(edges2,bins2,yerr=bins4,color="orange",fmt='',lw=1,label=label_incorrect)
    ax.scatter(edges2,bins2,s=5,color="orange")
    ax.set_ylabel('#')
    leg = ax.legend()

    if plot_type == '1-ring':
        ax.set_title("Predicted single rings")
    else:
        ax.set_title('Predicted multi rings')

    ax = fig.add_subplot(2,1,2)
    rat = getRatio(bins2,bins)
    error_rat = getRatioError(bins2,bins)
    plt.grid()
    ax.set_axisbelow(True)
    ax.errorbar(edges,rat,yerr=error_rat,fmt='none',l2=1,color='red')
    ax.scatter(edges,rat,s=5,color='red')
    #ax.set_ylim(min(rat)-0.05,max(rat)+0.05)
    ax.set_ylim(-0.05,1.1)
    ax.set_xlabel(varname)
    ax.set_ylabel("Precision")
                     
    if plot_type == '1-ring':
        plt.savefig("plots/RingClassification/EnergyDependence/RingClassification_Precision_"+dataset_name+"_"+variable_config+"_"+varname+"_single.pdf")
    else:
        plt.savefig("plots/RingClassification/EnergyDependence/RingClassification_Precision_"+dataset_name+"_"+variable_config+"_"+varname+"_multi.pdf")
    plt.close("all")

    for i in range(len(edges)):
        prec_file.write(plot_type+","+str(edges[i])+","+str(bins[i])+","+str(bins2[i])+","+str(rat[i])+","+str(error_rat[i])+"\n")

# ------------------------------------------------------------------------------------------
# ------ plot_histograms: Plot histograms for number of classified events (variable) -------
# ------------------------------------------------------------------------------------------

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
        plt.title("Misclassified 1-rings ("+dataset_name+" dataset)")
    elif plot_type == 'multi-ring':
        plt.title('Misclassified multi rings ('+dataset_name+' dataset)')
    else:
        plt.title('Misclassified events ('+dataset_name+' dataset)')
    plt.legend(loc='upper left')

    if plot_type == '1-ring':
        plt.savefig("plots/RingClassification/EnergyDependence/RingClassification_Classification_Hist_"+dataset_name+"_"+variable_config+"_"+varname+"_single.pdf")
    elif plot_type == 'multi-ring':
        plt.savefig("plots/RingClassification/EnergyDependence/RingClassification_Classification_Hist_"+dataset_name+"_"+variable_config+"_"+varname+"_multi.pdf")
    else:
        plt.savefig("plots/RingClassification/EnergyDependence/RingClassification_Classification_Hist_"+dataset_name+"_"+variable_config+"_"+varname+"_overall.pdf")
    plt.close("all")


plot_types = ['overall','1-ring','multi-ring']

print('Length list_plot_variables: ',len(list_plot_variables))
print('Length list_lowerbin: ',len(list_lowerbin))
print('Length list_upperbin: ',len(list_upperbin))
print('Length list_binwidth: ',len(list_binwidth))

for i in range(len(list_plot_variables)):

    correct.clear()
    correct_single.clear()
    correct_multi.clear()
    pred_correct.clear()
    pred_correct_single.clear()
    pred_correct_multi.clear()
    incorrect.clear()
    incorrect_single.clear()
    incorrect_multi.clear()
    pred_incorrect.clear()
    pred_incorrect_single.clear()
    pred_incorrect_multi.clear()

    if list_plot_variables[i] != "Pip":
        create_lists(list_plot_variables[i])

        for plottype in plot_types:
            plot_histograms(list_lowerbin[i],list_upperbin[i],list_binwidth[i],list_plot_variables[i],plottype)
            plot_ratios(list_lowerbin[i],list_upperbin[i],list_binwidth[i],list_plot_variables[i],plottype)
            if plottype == '1-ring' or plottype == 'multi-ring':
                plot_precision(list_lowerbin[i],list_upperbin[i],list_binwidth[i],list_plot_variables[i],plottype)

    else:
        create_lists_pip()
        plot_histograms(list_lowerbin[i],list_upperbin[i],list_binwidth[i],"PiPlus",'overall')
        plot_ratios(list_lowerbin[i],list_upperbin[i],list_binwidth[i],"PiPlus",'overall')

        correct.clear()
        incorrect.clear()

        create_lists_pim()
        plot_histograms(list_lowerbin[i],list_upperbin[i],list_binwidth[i],"PiMinus",'overall')
        plot_ratios(list_lowerbin[i],list_upperbin[i],list_binwidth[i],"PiMinus",'overall')

        correct.clear()
        incorrect.clear()

        create_lists_pi0()
        plot_histograms(list_lowerbin[i],list_upperbin[i],list_binwidth[i],"Pi0",'overall')
        plot_ratios(list_lowerbin[i],list_upperbin[i],list_binwidth[i],"Pi0",'overall')

        correct.clear()
        incorrect.clear()

        create_lists_kp()
        plot_histograms(list_lowerbin[i],list_upperbin[i],list_binwidth[i],"KPlus",'overall')
        plot_ratios(list_lowerbin[i],list_upperbin[i],list_binwidth[i],"KPlus",'overall')

        correct.clear()
        incorrect.clear()

        create_lists_km()
        plot_histograms(list_lowerbin[i],list_upperbin[i],list_binwidth[i],"KMinus",'overall')
        plot_ratios(list_lowerbin[i],list_upperbin[i],list_binwidth[i],"KMinus",'overall')

