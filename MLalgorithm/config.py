"""
@Description :   
@Author      :   Yang Xu
"""

import os
from skopt.space import Real, Categorical, Integer

##########Classificaiton Models##########
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
##########Classificaiton Models##########


Path = (os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/")
##################################################Config##################################################
eeg_file = Path + "Assets/dataset/EEG (EO, AC1, AC2).xlsx"
ecg_file = Path + "Assets/dataset/ECG (EO, AC1, AC2).xlsx"
Assets_folder = Path + "Assets/"
model_folder = Assets_folder + "models/"
plot_folder = Assets_folder + "plots/"
show_plots = False

features = ['Mean HR (BPM)', 'AVNN (ms)', 'SDNN (ms)', 'NN50 (beats)', 'pNN50 (%)', 'RMSSD (ms)', 'LF (ms2)', 'LF Norm (n.u.)', 'HF (ms2)', 'HF Norm (n.u.)', 'LF/HF Ratio', 'Fp 1', 'Fp 2', 'F 3', 'F 4', 'T 3', 'T 4', 'P 3', 'P 4', 'Fp 1', 'Fp 2', 'F 3', 'F 4', 'T 3', 'T 4', 'P 3', 'P 4', 'Fp 1', 'Fp 2', 'F 3', 'F 4', 'T 3', 'T 4', 'P 3', 'P 4', 'Fp 1', 'Fp 2', 'F 3', 'F 4', 'T 3', 'T 4', 'P 3', 'P 4', 'Fp 1', 'Fp 2', 'F 3', 'F 4', 'T 3', 'T 4', 'P 3', 'P 4', 'Fp 1', 'Fp 2', 'F 3', 'F 4', 'T 3', 'T 4', 'P 3', 'P 4', 'Fp 1', 'Fp 2', 'F 3', 'F 4', 'T 3', 'T 4', 'P 3', 'P 4']
target = 'Segment'

Model_params = {
    DecisionTreeClassifier:{
        'max_depth': Integer(2, 30),
        'min_samples_split': Integer(2, 30)
    },
    RandomForestClassifier:{
        'n_estimators': Integer(2, 100), 
        'min_samples_split': Integer(3, 25),
        'max_depth': Integer(1, 30)
	},
    GradientBoostingClassifier:{
        'n_estimators': Integer(2, 100),
        'max_depth': Integer(2, 30),
        'learning_rate': Real(1e-4, 1, prior = 'log-uniform')
    },
    SVC:{
        'kernel':Categorical(['rbf', 'linear']),
        'C':Real(0.1, 10),
        'probability':Categorical([True])
    },
    MLPClassifier:{
        'learning_rate_init': Real(1e-5, 0.5,prior = 'log-uniform'),
        'hidden_layer_sizes': Integer(2, 250)
    }
}
##################################################Config##################################################
