import os, pickle
import config, test_data_generator
import tqdm as tqdm


models = ['DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'SVC', 'MLPClassifier']
model = models[4]
#
Path = (os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/")
os.chdir(Path)


def run(input):
    with open(config.model_folder + 'Scaler.pkl','rb') as f:
        scaler = pickle.load(f)
    with open(config.model_folder + 'LBE.pkl','rb') as f:
        lbe = pickle.load(f)
    with open(config.model_folder + 'PCA.pkl','rb') as f:
        pca = pickle.load(f)
    with open(config.model_folder + model + '.pkl','rb') as f:
        trained_model = pickle.load(f)
        
    input = scaler.transform(input)
    input = pca.transform(input)
    y_pred = trained_model.predict(input)
    pred_label = lbe.inverse_transform(y_pred)
    
    print(f"Prediction Result: {pred_label}")
    return pred_label



if __name__ == "__main__":
    iters = 100
    rdm_seed = [i for i in range(iters)]
    for i in tqdm.tqdm(range(iters)):
        input = test_data_generator.generate(rdm_seed[i])
        run(input)

  