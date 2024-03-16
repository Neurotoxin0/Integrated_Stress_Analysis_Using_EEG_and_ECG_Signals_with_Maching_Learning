import os, pickle, random, sys, time, warnings
import config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV


Path = (os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/")
sys.path.append(Path)
os.chdir(Path)

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)
np.int = np.int32
#sns.set()


def main():
    # EEG data
    data_eeg = pd.read_excel(config.eeg_file, header = None)
    data_eeg = data_eeg.fillna(method = 'ffill')  # fill the NaN with the previous non-NaN value
    data_eeg.columns = data_eeg.iloc[1]
    data_eeg = data_eeg.iloc[2:]    # remove the first two rows
    #data_eeg = data_eeg.iloc[:,:11]

    # ECG data
    data_ecg_EO = pd.read_excel(config.ecg_file,'EO')
    data_ecg_EO['Segment'] = 'EO'
    data_ecg_AC1 = pd.read_excel(config.ecg_file,'AC1')
    data_ecg_AC1['Segment'] = 'AC1'
    data_ecg_AC2 = pd.read_excel(config.ecg_file,'AC2')
    data_ecg_AC2['Segment'] = 'AC2'
    dfs = [data_ecg_EO, data_ecg_AC1, data_ecg_AC2]
    dfs = [dt.rename(columns = {"Mean HR (bpm)":'Mean HR (BPM)'}) for dt in dfs]
    data_ecg = pd.concat(dfs, axis = 0).reset_index(drop = True).rename(columns = {'Subject NO.':"Subject (No.)"})     # combine the three dataframes, reset the index and rename the column
    #data_ecg   # debug

    # merge ECG and EEG data
    data = pd.merge(data_ecg, data_eeg, on = ['Segment', 'Subject (No.)', 'Gender'])
    #print(data.columns.tolist())  # debug

    # standardize the data
    scaler = StandardScaler()   # standardize the data to have a mean of 0 and a standard deviation of 1
    X = data.drop(columns=['Segment', 'Subject (No.)', 'Gender'])
    X = scaler.fit_transform(X)
    d = pickle.dumps(scaler)
    if not os.path.exists(config.model_folder): os.makedirs(config.model_folder)
    with open(config.model_folder + 'Scaler.pkl', 'wb+') as f: f.write(d)  # wb+ -> write binary

    # encode the target variable and save the encoder
    lbe = LabelEncoder()
    data[config.target] = lbe.fit_transform(data[config.target])   
    y = data[config.target]    # target variable
    d = pickle.dumps(lbe)
    if not os.path.exists(config.model_folder): os.makedirs(config.model_folder)
    with open(config.model_folder + 'LBE.pkl', 'wb+') as f: f.write(d)

    # perform PCA
    PCA_n(X)

    # save PCA model
    X_pca, pca = pcaX(X, 0.95)
    d = pickle.dumps(pca)
    if not os.path.exists(config.model_folder): os.makedirs(config.model_folder)
    with open(config.model_folder + 'PCA.pkl', 'wb+') as f: f.write(d)
        
    # optimization
    best_params_dic = {}
    best_models = []
    for Model, params in config.Model_params.items():
        opt, best_params_, best_score_ = bayes_opt(Model, params, X_pca, y, n_iter = 10, scoring = 'f1_weighted', cv = 3)
        best_params_dic[Model] = dict(best_params_)
        best_models.append(Model(**dict(best_params_)))

    ## train test split 80%/20%     #TODO: use Cross-Validation
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size = 0.20, random_state = 42, shuffle = True)

    # train the models
    models_untrained = best_models.copy()
    model_type = 'classification'
    eval_dic = {}
    eval_dic_train = {} # Training Set Evaluation Results
    models = []

    for model_untrained in models_untrained:
        model_name = type(model_untrained).__name__
        print(f"[{model_name}]")
        model, metrics_train, metrics_test = model_train(model_untrained, X_train, y_train, X_test = X_test, y_test = y_test, model_type = model_type)
        models.append(model)
        
        # Get Test Set Evaluation Results
        eval_dic[model_name] = metrics_test.copy()
        # Get Training Set Evaluation Results
        eval_dic_train[model_name] = metrics_train.copy()

        print("Evaluation on Training Data:")
        print(metrics_train)
        print("Evaluation on Test Data:")
        print(metrics_test)

        # save model
        d = pickle.dumps(model)
        if not os.path.exists(config.model_folder): os.makedirs(config.model_folder)
        with open(config.model_folder + model_name + '.pkl','wb+') as f: f.write(d)
        print('\n')

    eval_df = plot_evaluation_comparison(eval_dic)

    # find the best model
    metric = 'F1-Score'
    model_perf = eval_df[eval_df['Metric'] == metric].sort_values(by = 'Value', ascending=False)
    best_score = model_perf.iloc[0]['Value']
    best_model_name = model_perf.iloc[0]['Model']
    print(f"\n\nBest model is {best_model_name}, {metric} is {best_score:.4f}")

    for model in models:
        if type(model).__name__ == best_model_name:
            best_model = model
            print(f"Best model: {best_model}\n\n")
            break

    # Cross Validation Scores  
    model_CrossValScoresPlot(models, X, y, scoring = 'accuracy', cv = 5)


def PCA_n(X_scaled):
    """
    Perform Principal Component Analysis (PCA) and plot the cumulative explained variance ratio.

    Parameters:
    - X_scaled: numpy array, shape (n_samples, n_features)
                Scaled feature matrix.
    """
    estimator_pca = PCA(n_components = None)
    estimator_pca.fit(X_scaled)     # Fit PCA model to the scaled feature matrix
    evr = estimator_pca.explained_variance_ratio_ 

    # Plot the cumulative explained variance ratio
    plt.figure(figsize = (8, 5))
    plt.plot(np.arange(1, len(evr) + 1), np.cumsum(evr * 100), "-o")
    plt.title("Cumulative Explained Variance Ratio", fontsize = 15)
    plt.xlabel("number of components", fontsize = 15)
    plt.ylabel("explained variance ratio(%)", fontsize = 15)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    if config.show_plots: plt.show()


def pcaX(X_scaled, n):
    """
    Perform Principal Component Analysis (PCA) and reduce the dimensionality of the feature matrix.

    Parameters:
    - X_scaled: numpy array, shape (n_samples, n_features)
                Scaled feature matrix.
    - n: int
            Number of principal components to retain.

    Returns:
    - pca_X: pandas DataFrame, shape (n_samples, n)
                Transformed feature matrix with reduced dimensionality.
    - pca: PCA object
            Fitted PCA model.
    """
    pca = PCA(n_components = n)
    pca_X = pca.fit_transform(X_scaled)
    print(f"Decomposition: {X_scaled.shape} --> {pca_X.shape}")   # debug
    return pd.DataFrame(pca_X), pca


# optimization method
def bayes_opt(Model,params:dict, X, y, n_iter = 5, scoring = 'r2', cv = 3):
    """
    Perform Bayesian optimization for hyperparameter tuning of a given model.

    Parameters:
    - Model: class
                Machine learning model class (e.g., RandomForestRegressor, XGBRegressor).
    - params: dictionary
                Dictionary containing hyperparameter ranges to be searched.
    - X: numpy array or pandas DataFrame, shape (n_samples, n_features)
            Feature matrix.
    - y: numpy array or pandas Series, shape (n_samples,)
            Target vector.
    - n_iter: int, default=5
                Number of parameter settings that are sampled.
    - scoring: str, default='r2'
                Scoring metric to optimize during cross-validation.
    - cv: int or cross-validation generator, default=3
            Number of folds for cross-validation.

    Returns:
    - opt: BayesSearchCV object
            Fitted BayesSearchCV object.
    - best_params: dict
                    Best hyperparameters found during optimization.
    - best_score: float
                    Best score achieved during optimization.
    """
    print(f"- Bayes Optimizing: {Model.__name__}")
    opt = BayesSearchCV(
        Model(),
        params,
        n_iter = n_iter,
        scoring = scoring,
        cv = cv,
        verbose = 1
    )
    opt.fit(X, y)
    #print(opt.best_params_)
    #print(opt.best_score_)
    return opt, opt.best_params_, opt.best_score_


def evaluation(model = None, X_test = None, y_test = None, y_pred = None, model_type = None):
    """
    Evaluate the performance of a model given the true values (y_test) and predicted values (y_pred).

    Parameters:
    - model: object (optional)
                Trained machine learning model.
    - X_test: numpy array or pandas DataFrame (optional), shape (n_samples, n_features)
                Test feature matrix.
    - y_test: numpy array or pandas Series (optional), shape (n_samples,)
                True labels for the test set.
    - y_pred: numpy array (optional), shape (n_samples,)
                Predicted labels or values for the test set.
    - model_type: str (optional)
                    Type of model ('classification' or 'regression').

    Returns:
    - metrics: dictionary
                Dictionary containing evaluation metrics.
    """
    if model is not None: y_pred = model.predict(X_test)

    # check if one-hot encoded:
    if len(y_pred.shape) == 2: y_pred = y_pred.argmax(axis=1)
    if len(y_test.shape) == 2: y_test = y_test.argmax(axis=1)

    metrics = {}
    # if classification model
    if model_type == 'classification':
        from sklearn.metrics import classification_report
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score
        # metrics['cls_report'] = classification_report(y_test, y_pred)
        metrics['accuracy']     = accuracy_score(y_test,y_pred)
        metrics['precision']    = precision_score(y_test,y_pred,average='macro')
        metrics['recall']       = recall_score(y_test,y_pred,average='macro')
        metrics['f1-score']     = f1_score(y_test,y_pred,average='macro')
        
    # if regression model
    elif model_type == 'regression':
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import r2_score
        metrics['MSE']  = mean_squared_error(y_test,y_pred)
        metrics['RMSE'] = mean_squared_error(y_test,y_pred) ** 0.5
        metrics['MAE']  = mean_absolute_error(y_test,y_pred)
        metrics['R2']   = r2_score(y_test,y_pred)

    else:
        raise Exception('model_type should be classification or regression!')

    return metrics


def model_train(model, X_train, y_train, X_test = None, y_test = None, model_type = None):
    """
    Train a machine learning model and evaluate its performance.

    Parameters:
    - model: object
                Machine learning model to be trained.
    - X_train: numpy array or pandas DataFrame, shape (n_samples, n_features)
                Training feature matrix.
    - y_train: numpy array or pandas Series, shape (n_samples,)
                Training labels or target values.
    - X_test: numpy array or pandas DataFrame (optional), shape (n_samples, n_features)
                Test feature matrix.
    - y_test: numpy array or pandas Series (optional), shape (n_samples,)
                Test labels or target values.
    - model_type: str (optional)
                    Type of model ('classification' or 'regression').

    Returns:
    - model: object
                Trained machine learning model.
    - metrics_train: dict or None
                        Evaluation metrics on the training set.
    - metrics_test: dict or None
                    Evaluation metrics on the test set.
    """
    time_start = time.time()
    model.fit(X_train, y_train)
    time_end = time.time()
    time_cost = time_end - time_start
    print(f"- training {type(model).__name__}, time cost: {round(time_cost)} seconds")

    metrics_test = None
    metrics_train = None
    if X_test is not None:
        y_pred_test = model.predict(X_test)
        metrics_test = evaluation(X_test = X_test, y_test = y_test, y_pred = y_pred_test, model_type = model_type)
        y_pred_train = model.predict(X_train)
        metrics_train = evaluation(X_test = X_test, y_test = y_train, y_pred = y_pred_train, model_type = model_type)
    
    return model, metrics_train, metrics_test



def plot_evaluation_comparison(eval_dic, metric_x = 'Metric', metric_hue = 'Model', dataset_name = None):
    """
    Plot a comparison of evaluation metrics for different models.

    metric_dic looks like: {'model_name':{'metric_name': value}}

    Parameters:
    - eval_dic: dictionary
                Dictionary containing evaluation metrics for different models.
    - metric_x: str, default='Metric'
                Name of the metric to be plotted on the x-axis.
    - metric_hue: str, default='Model'
                    Name of the metric used for coloring (hue) the bars.
    - dataset_name: str or None, default=None
                    Name of the dataset (optional).

    Returns:
    - eval_df: pandas DataFrame
                DataFrame containing the evaluation metrics for plotting.
    """
    # converts to pd.dataframe
    eval_df = pd.DataFrame([[md, mt.title(), v] for md, dic in eval_dic.items() for mt, v in dic.items()],
                            columns= ['Model','Metric','Value'])
    eval_df.sort_values(by=['Metric', 'Value'], inplace=True)
    eval_df.reset_index(drop = True, inplace = True)
    print(eval_df)

    # plot
    plt.figure(figsize = (10,7))
    sns.barplot(data = eval_df, x = metric_x, y = 'Value', hue = metric_hue)
    plt.title(f"Model Comparison", fontsize = 15)
    plt.xticks(rotation = 0)
    plt.ylim(eval_df['Value'].min() * 0.8, eval_df['Value'].max() * 1.05)
    plt.legend(loc = 0, prop = {'size':8})
    plt.tight_layout()
    if not os.path.exists(config.plot_folder): os.makedirs(config.plot_folder)
    plt.savefig(f"{config.plot_folder}" + (f"{dataset_name} - " if dataset_name else '') + f"Comparison.jpg", dpi = 300)
    if config.show_plots: plt.show()

    return eval_df



def model_CrossValScoresPlot(models:list, X, y, scoring = 'accuracy', cv = 3, plots_type = ['boxplot', 'barplot']):
    """
    Plot cross-validation scores for multiple models.

    Parameters:
    - models: list
                List of unfitted models with parameters, e.g., [Model(param=1)].
    - X: numpy array or pandas DataFrame, shape (n_samples, n_features)
            Feature matrix.
    - y: numpy array or pandas Series, shape (n_samples,)
            Target vector.
    - scoring: str, default='accuracy'
                Scoring method for cross-validation.
    - cv: int or cross-validation generator, default=3
            Number of folds for cross-validation.
    - plots_type: list of str, default=['boxplot', 'barplot']
                    Types of plots to generate ('boxplot' and/or 'barplot').

    Returns:
    - df_score: pandas DataFrame
                DataFrame containing cross-validation scores for each model.
    """
    score_dic = {}
        
    for model in models:
        print(f"- Cross validation: {type(model).__name__}")
        score_dic[type(model).__name__] = cross_val_score(model, X, y, scoring = scoring, cv = cv)
    df_score = pd.DataFrame(score_dic)

    # box plot
    if 'boxplot' in plots_type:
        plt.figure(figsize = (12, 6))
        plt.boxplot([df_score[col] for col in df_score.columns])
        s = pd.concat([df_score[col].rename("scoring") for col in df_score.columns], axis=0)
        plt.ylim(s.min() * 0.7, s.max() * 1.2)
        plt.xticks(range(1, len(df_score.columns) + 1), df_score.columns)
        plt.ylabel(scoring)
        plt.xlabel("Models")
        plt.title("Model Comparison - Cross Validation")
        if not os.path.exists(config.plot_folder): os.makedirs(config.plot_folder)
        plt.savefig(f"{config.plot_folder} + Model Comparison - Cross Validation.jpg",dpi=300)
        if config.show_plots: plt.show()

    # bar plot
    if 'barplot' in plots_type:
        score_mean = df_score.mean(axis = 0)
        plt.figure(figsize = (12, 6))
        x = score_mean.index
        y = score_mean
        plt.bar(x, y)
        for i, j in zip(range(len(x)), y):
            plt.text(i, j, '{:.4}'.format(j), va = 'bottom', ha = 'center')
        plt.ylabel(scoring)
        plt.xlabel("Models")
        plt.ylim(y.min() * 0.7, y.max() * 1.2)
        plt.title(f"Model Comparison - {scoring}")
        plt.savefig(f"{config.plot_folder} + Model Comparison - {scoring}.jpg", dpi = 300)
        if config.show_plots: plt.show()
            
    return df_score



if __name__ == "__main__":
    main()

