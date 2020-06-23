# Classification - ANNIE #

The two classification tasks in ANNIE are 
* RingClassification: The identification of single-ring events (events with a single Cherenkov ring).
* Particle ID (PID): The distinction of muon from electron events.

## Scripts ##

The scripts in this directory exist to perform these classification tasks with the help of ML classifiers such as the Random Forest, Support Vector Machine (SVM), Support Gradient Descent (SGD), Multi-Layer Perceptron (MLP), GradientBoosting and XGBoost.

The different classification scripts do the following:
* `do_classification_emu.py`: Use the trained model to perform electron/muon discrimination on a new dataset in csv-format. The script checks whether the model and the dataset contain the same variables. The output is a csv-file with two columns, containing the predicted class and the predicted probability.
* `do_classification_rings.py`: Use the trained model to perform single-/multi-ring discrimination on a new dataset in csv-format. The script checks whether the model and the dataset contain the same variables. The output is a csv-file with two columns, containing the predicted class and the predicted probability.
* `eval_misclass_emu.py`: Evaluate the misclassification rate of the PID as a function of different parameters, such as the muon/electron energy. The list of parameters to be used to create the plots can be customly defined.
* `eval_misclass_rings.py`: Evaluate the misclassification rate of the RingClassification as a function of different parameters, such as the neutrino/muon energy. The list of parameters to be used to create the plots can be customly defined.
* `eval_model_emu.py`: Evaluate a PID model performance on a dataset by comparing the predicted with the true class labels.
* `eval_model_rings.py`: Evaluate a RingClassification model performance on a dataset by comparing the predicted with the true class labels.
* `optimising_parameters_emu.py`: Scan through a grid of hyperparameters to find optimum PID classifier configuration.
* `optimising_parameters_rings.py`: Scan through a grid of hyperparameters to find optimum RingClassification classifier configuration.
* `plot_cmatrix_emu.py`: Plot confusion matrix for PID classifiers.
* `plot_cmatrix_rings.py`: Plot confusion matrix for RingClassification classifiers.
* `plot_purity_emu.py`: Plot purity/efficiency/accuracy curves for PID classifiers.
* `plot_purity_rings.py`: Plot purity/efficiency/accuracy curves for RingClassification classifiers.
* `rank_variables_emu.py`: Rank variables for PID classifiers.
* `rank_variables_rings.py`: Rank variables for RingClassification classifiers.
* `train_classification_emu.py`: Train PID classifiers on data. The models are saved in the `models` directory.
* `train_classification_rings.py`: Train RingClassification classifiers on data. The models are saved in the `models` directory.

## Directory structure ##

The directory structure is the following:
* `accuracy`: Text files containing the overall accuracies for different classifiers on different datasets.
* `additional_event_info.nosync`: Additional event information which is used to create the plots for `eval_misclass_*py`.
* `data_new.nosync`: New data files [csv-format]. Data files are created with the `ToolAnalysis` framework based on `WCSim` simulation files. (see below)
* `data_old.nosync`: Older data files [csv-format]. Data files are created with the `ToolAnalysis` framework based on `WCSim` simulation files. (see below)
* `examples`: Some basic example classifiers and datasets.
* `featurevars`: Distributions of the featured classification variables, comparison between the different datasets.
* `hyperparameters`: Results of hyperparameter scans.
* `indices`: Indices of the events that are stored in the test sets (in case one wants to look at specific events)
* `models`: Trained models are saved in this directory.
* `plots`: All plots are saved in this directory. Subdirectories are:
  * `ConfusionMatrix`: Confusion matrices
  * `Correlation`: 2D-scatter-plots of classification variables. Different classes are colored differently.
  * `EnergyDependence`: Accuracy/purity dependence on the energy of primary particles/secondary particles. 
  * `FeatureImportance`: Importance ranking of the classification variables.
  * `PredProbability`: Distributions of the predicted probabilities, purity/efficiency/rel. uncertainty curves.
  * `ROC`: ROC & Precision-Recall curves.
* `predictions`: Predictions of the classifiers for the test datasets together with the true class labels.
* `variable_config`: Different variable configuration sets. The classifier uses the variables defined in this set to perform the training and later the classification.

## Classification variables / data ##

The classification variables are computed within the ToolAnalysis framework (https://github.com/ANNIEsoft/ToolAnalysis.git), specifically in the tools `CalcClassificationVars` and `StoreClassificationVars`. The variables try to reflect meaningful properties of the event such as angular/time/charge distribution properties.
The tools produce labeled csv-output files containing the classification variables that can be used to train the classifiers with the scripts here.
