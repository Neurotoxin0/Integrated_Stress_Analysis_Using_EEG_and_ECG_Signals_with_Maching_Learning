"""
@Description :   
@Author      :   Yang Xu
Modified by Xilai Wang to enable input from and output to the Django back-end server.
"""

import os, pickle
import config, test_data_generator
import tqdm as tqdm
import argparse
import json
import pandas as pd
import config
import numpy as np


Path = (os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/")
os.chdir(Path)

#=======Section 1: accepting arguments from Django server and pre-process them. Written by Xilai Wang=========== 
#get the model selected and the input features for ML model.
parser = argparse.ArgumentParser(description='Run the Machine Learning models and return the prediction results.')
parser.add_argument('json_param', type=str, help='A JSON string containing the model name (string) and input features (float[67]).')
args = parser.parse_args()
params = json.loads(args.json_param)
model_name = params["model_name"]     # one of mlp, svc, gbc, dtc, rfc. 
input_features = params["input_features"]    # float[67]

models = ['DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'SVC', 'MLPClassifier']
if model_name == "mlp":
	model = models[4]
elif model_name == "svc":
	model = models[3]
elif model_name == "gbc":
	model = models[2]
elif model_name == "dtc":
	model = models[0]
elif model_name == "rfc":
	model = models[1]

#=======Section 2: Running model to do prediction and output the result. Written by Yang Xu.=========== 
#model = 'MLPClassifier'   # manual debug usage
with open(config.model_folder + 'Scaler.pkl','rb') as f:
    scaler = pickle.load(f)
with open(config.model_folder + 'LBE.pkl','rb') as f:
    lbe = pickle.load(f)
with open(config.model_folder + 'PCA.pkl','rb') as f:
    pca = pickle.load(f)
with open(config.model_folder + model + '.pkl','rb') as f:
    trained_model = pickle.load(f)


#input = test_data_generator.generate() # generate a random input for testing
input = [input_features]
input = pd.DataFrame(input, columns = config.features)
input = scaler.transform(input)
input = pca.transform(input)
y_pred = trained_model.predict(input)
pred_label = lbe.inverse_transform(y_pred)
print(pred_label)    
    