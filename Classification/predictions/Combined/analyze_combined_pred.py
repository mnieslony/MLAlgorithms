import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


predictions = pd.read_csv("RingClassification_PID_MLP_predictions_beam_PMTOnly_Q.csv",header=0)

truth = predictions["MCMultiRing"]
ringpred = predictions["RingPrediction"]
pidtruth = predictions["particleType"]
pidpred = predictions["PIDPrediction"]
pmtq = predictions["PMTQtotal"]

n_single_ring = 0
n_multi_ring = 0
n_muon = 0
n_electron = 0
n_single_ring_muon = 0
n_multi_ring_muon = 0
n_single_ring_electron = 0
n_multi_ring_electron = 0

n_single_ring_correct = 0
n_multi_ring_correct = 0
n_single_ring_muon_correct = 0
n_single_ring_electron_correct = 0
n_electron_correct = 0

n_muSR_muSR = 0
n_muSR_muMR = 0
n_muSR_eSR = 0
n_muSR_eMR = 0
n_muMR_muSR = 0
n_muMR_muMR = 0
n_muMR_eSR = 0
n_muMR_eMR = 0
n_eSR_eSR = 0
n_eSR_eMR = 0
n_eSR_muSR = 0
n_eSR_muMR = 0
n_eMR_muSR = 0
n_eMR_muMR = 0
n_eMR_eSR = 0
n_eMR_eMR = 0

charge_true_muSR = []
charge_pred_muSR = []
charge_true_total = []
charge_true_muMR = []

for i in range(len(truth)):

	charge_true_total.append(pmtq[i])
	if truth[i] == "1-ring":
		n_single_ring+=1
		if ringpred[i] == "1-ring":
			n_single_ring_correct+=1
		if pidtruth[i] == "muon":
			n_single_ring_muon += 1
			charge_true_muSR.append(pmtq[i])
			if ringpred[i] == "1-ring" and pidpred[i] == "muon":
				n_single_ring_muon_correct +=1
				n_muSR_muSR += 1
				charge_pred_muSR.append(pmtq[i])
			elif ringpred[i] == "1-ring" and pidpred[i] == "electron":
				n_muSR_eSR += 1
			elif ringpred[i] == "multi-ring" and pidpred[i] == "muon":
				n_muSR_muMR += 1
			elif ringpred[i] == "multi-ring" and pidpred[i] == "electron":
				n_muSR_eMR += 1
		else:
			n_single_ring_electron += 1
			if ringpred[i] == "1-ring" and pidpred[i] == "electron":
				n_single_ring_electron_correct +=1
				n_eSR_eSR += 1
			elif ringpred[i] == "1-ring" and pidpred[i] == "muon":
				n_eSR_muSR +=1
				charge_pred_muSR.append(pmtq[i])
			elif ringpred[i] == "multi-ring" and pidpred[i] == "electron":
				n_eSR_eMR +=1
			elif ringpred[i] == "multi-ring" and pidpred[i] == "muon":
				n_eSR_muMR +=1
			if pidpred[i] == "electron":
				n_electron_correct +=1
	else:
		n_multi_ring+=1
		if ringpred[i] == "multi-ring":
			n_multi_ring_correct +=1
		if pidtruth[i] == "muon":
			n_multi_ring_muon += 1
			charge_true_muMR.append(pmtq[i])
			if ringpred[i] == "multi-ring" and pidpred[i] == "muon":
				n_muMR_muMR += 1
			elif ringpred[i] == "multi-ring" and pidpred[i] == "electron":
				n_muMR_eMR += 1
			elif ringpred[i] == "1-ring" and pidpred[i] == "muon":
				n_muMR_muSR += 1
				charge_pred_muSR.append(pmtq[i])
			elif ringpred[i] ==  "1-ring" and pidpred[i] == "electron":
				n_muMR_eSR += 1
		else:
			n_multi_ring_electron += 1
			if ringpred[i] == "multi-ring" and pidpred[i] == "electron":
				n_eMR_eMR += 1
			elif ringpred[i] == "multi-ring" and pidpred[i] == "muon":
				n_eMR_muMR += 1
			elif ringpred[i] == "1-ring" and pidpred[i] == "electron":
				n_eMR_eSR += 1
			elif ringpred[i] == "1-ring" and pidpred[i] == "muon":
				n_eMR_muSR += 1
				charge_pred_muSR.append(pmtq[i])


print("Prediction summary (Full dataset):")

print("Single rings: "+str(n_single_ring)+" (fraction: "+str(n_single_ring/(n_single_ring+n_multi_ring))+")")
print("Multi rings: "+str(n_multi_ring)+" (fraction: "+str(n_multi_ring/(n_single_ring+n_multi_ring))+")")
print("Single rings (muon): "+str(n_single_ring_muon)+" (fraction: "+str(n_single_ring_muon/(n_single_ring+n_multi_ring))+")")
print("Single rings (electron): "+str(n_single_ring_electron)+" (fraction: "+str(n_single_ring_electron/(n_single_ring+n_multi_ring))+")")
print("Multi rings (muon): "+str(n_multi_ring_muon)+" (fraction: "+str(n_multi_ring_muon/(n_single_ring+n_multi_ring))+")")
print("Multi rings (electron): "+str(n_multi_ring_electron)+" (fraction: "+str(n_multi_ring_electron/(n_single_ring+n_multi_ring))+")")

print("---------------------------------------------------------------")
print("Correct single rings: "+str(n_single_ring_correct)+" (fraction :"+str(n_single_ring_correct/n_single_ring)+")")
print("Correct muti rings: "+str(n_multi_ring_correct)+" (fraction :"+str(n_multi_ring_correct/n_multi_ring)+")")
print("Correct single muon rings: "+str(n_single_ring_muon_correct)+" (fraction :"+str(n_single_ring_muon_correct/n_single_ring_muon)+")")
print("Correct single electron rings: "+str(n_single_ring_electron_correct)+" (fraction :"+str(n_single_ring_electron_correct/n_single_ring_electron)+")")
print("Correct electron rings: "+str(n_electron_correct)+" (fraction :"+str(n_electron_correct/n_single_ring_electron)+")")

print("---------------------------------------------------------------")
print("Muon single ring:")
print("Total muon single rings: "+str(n_single_ring_muon))
print("Predicted: muSR: "+str(n_muSR_muSR)+", fraction: "+str(n_muSR_muSR/n_single_ring_muon))
print("Predicted: muMR: "+str(n_muSR_muMR)+", fraction: "+str(n_muSR_muMR/n_single_ring_muon))
print("Predicted: eSR: "+str(n_muSR_eSR)+", fraction: "+str(n_muSR_eSR/n_single_ring_muon))
print("Predicted: eMR: "+str(n_muSR_eMR)+", fraction: "+str(n_muSR_eMR/n_single_ring_muon))

print("---------------------------------------------------------------")
print("Muon multi ring:")
print("Total muon single rings: "+str(n_multi_ring_muon))
print("Predicted: muSR: "+str(n_muMR_muSR)+", fraction: "+str(n_muMR_muSR/n_multi_ring_muon))
print("Predicted: muMR: "+str(n_muMR_muMR)+", fraction: "+str(n_muMR_muMR/n_multi_ring_muon))
print("Predicted: eSR: "+str(n_muMR_eSR)+", fraction: "+str(n_muMR_eSR/n_multi_ring_muon))
print("Predicted: eMR: "+str(n_muMR_eMR)+", fraction: "+str(n_muMR_eMR/n_multi_ring_muon))

print("---------------------------------------------------------------")
print("Electron single ring:")
print("Total electron single rings: "+str(n_single_ring_electron))
print("Predicted: muSR: "+str(n_eSR_muSR)+", fraction: "+str(n_eSR_muSR/n_single_ring_electron))
print("Predicted: muMR: "+str(n_eSR_muMR)+", fraction: "+str(n_eSR_muMR/n_single_ring_electron))
print("Predicted: eSR: "+str(n_eSR_eSR)+", fraction: "+str(n_eSR_eSR/n_single_ring_electron))
print("Predicted: eMR: "+str(n_eSR_eMR)+", fraction: "+str(n_eSR_eMR/n_single_ring_electron))

print("---------------------------------------------------------------")
print("Electron multi ring:")
print("Total electron multi rings: "+str(n_multi_ring_electron))
print("Predicted: muSR: "+str(n_eMR_muSR)+", fraction: "+str(n_eMR_muSR/n_multi_ring_electron))
print("Predicted: muMR: "+str(n_eMR_muMR)+", fraction: "+str(n_eMR_muMR/n_multi_ring_electron))
print("Predicted: eSR: "+str(n_eMR_eSR)+", fraction: "+str(n_eMR_eSR/n_multi_ring_electron))
print("Predicted: eMR: "+str(n_eMR_eMR)+", fraction: "+str(n_eMR_eMR/n_multi_ring_electron))

print("---------------------------------------------------------------")
print("Plot energy histograms")
bins_var = np.arange(0,5000,50)
#plt.hist(charge_true_total,bins_var, label='All CC events (FV)')
#plt.hist(charge_true_muMR,bins_var,alpha = 0.5, color = 'green', label='true muon MR')
plt.hist(charge_true_muSR,bins_var, label='true muon SR')
plt.hist(charge_pred_muSR,bins_var,alpha = 0.5, label='pred muon SR')
plt.xlabel('charge [p.e.]')
plt.ylabel('# events')
plt.title('ANNIE Beam test sample')
plt.legend()
plt.show()


