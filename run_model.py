import os, pickle
import config, test_data_generator
import numpy as np


Path = (os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/")
os.chdir(Path)


models = ['DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'SVC', 'MLPClassifier']
model = models[4]

'''
inputs = np.array(
	[[-0.17187095,  0.03125514,  0.1973567 ,  0.19280349,  0.15937716,
        0.14293361, -0.20381281, -0.73818189,  0.19498868,  0.7425749 ,
       -0.72619427, -0.42089305, -0.39533783,  0.04052541, -0.33819468,
        0.07512008, -0.64884972, -0.21106719, -0.4052345 , -0.31425958,
       -0.54230375,  0.08456723, -0.28505514,  0.41196632, -0.63870118,
       -0.04862251, -0.32219788, -0.33639321, -0.59832765,  0.082081  ,
       -0.20493673,  0.49241935, -0.40552546, -0.25007397, -0.43357805,
        0.02921497,  0.30828473,  1.26046425,  1.09774553,  0.5766993 ,
        0.48453208,  1.25080668,  0.62420446, -0.72795554, -0.68326625,
       -0.48345738, -0.53673692, -0.4976349 , -0.17072209, -0.01922287,
       -0.27699431, -0.79190093, -0.67586627, -0.32900944, -0.38265025,
       -0.43552977,  0.21048657, -0.26441517, -0.22625677, -0.89051863,
       -0.71512856, -0.27808073, -0.40093892, -0.41437903,  0.41723737,
       -0.38110601, -0.33821837]]
)
'''

# use the test_data_generator to generate a random input
inputs = test_data_generator.generate()

with open(config.model_folder + 'LBE.pkl','rb') as f:
	lbe = pickle.load(f)
	
with open(config.model_folder + 'PCA.pkl','rb') as f:
	pca = pickle.load(f)
	
with open(config.model_folder + model + '.pkl','rb') as f:
	model = pickle.load(f)

pca_input = pca.transform(inputs)
y_pred = model.predict(pca_input)
pred_label = lbe.inverse_transform(y_pred)
print(f"Prediction Result: {pred_label}")
