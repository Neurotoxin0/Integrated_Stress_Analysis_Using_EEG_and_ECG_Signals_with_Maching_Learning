import os, pickle, random, sys, time, warnings
import config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
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
#sns.set()


def main():
    # EEG data
    data_eeg = pd.read_excel(config.eeg_file,header=None)
    data_eeg = data_eeg.fillna(method='ffill')
    data_eeg.columns = data_eeg.iloc[1]
    data_eeg = data_eeg.iloc[2:]
    # data_eeg = data_eeg.iloc[:,:11]

    # ECG data
    data_ecg_EO = pd.read_excel(config.ecg_file,'EO')
    data_ecg_EO['Segment'] = 'EO'
    data_ecg_AC1 = pd.read_excel(config.ecg_file,'AC1')
    data_ecg_AC1['Segment'] = 'AC1'
    data_ecg_AC2 = pd.read_excel(config.ecg_file,'AC2')
    data_ecg_AC2['Segment'] = 'AC2'
    dfs = [data_ecg_EO,data_ecg_AC1,data_ecg_AC2]
    dfs = [dt.rename(columns = {"Mean HR (bpm)":'Mean HR (BPM)'}) for dt in dfs]
    data_ecg = pd.concat(dfs,axis=0).reset_index(drop=True).rename(columns={'Subject NO.':"Subject (No.)"})
    #data_ecg

    # merge ECG and EEG data
    data = pd.merge(data_ecg,data_eeg,on=['Segment', 'Subject (No.)','Gender'])

    # data preprocessing
    scaler = StandardScaler()
    X = data.drop(columns=['Segment','Subject (No.)','Gender'])
    X = scaler.fit_transform(X)
    y = data[config.target]

    lbe = LabelEncoder()
    data[config.target] = lbe.fit_transform(data[config.target])
	
    PCA_n(X)
	
    # save pca model
    X_pca,pca = pcaX(X,0.95)
    s = pickle.dumps(pca)
    if not os.path.exists(config.model_folder): os.makedirs(config.model_folder)
    with open(config.model_folder + 'PCA.pkl','wb+') as f:#注意此处mode是'wb+'，表示二进制写入
        f.write(s)
		
    # optimization
    best_params_dic = {}
    best_models = []
    for Model,params in config.Model_params.items():
        opt,best_params_,best_score_ = bayes_opt(Model,params, X_pca, y, n_iter = 10, scoring = 'f1_weighted', cv = 3)
        best_params_dic[Model] = dict(best_params_)
        best_models.append(Model(**dict(best_params_)))

    ## train test split 80%/20%
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.20, random_state = 42, shuffle = True)
	
    models_untrained = best_models.copy()

    model_type = 'classification'
    eval_dic = {}
    eval_dic_train = {} # 新增一个字典用于存储训练集的评估结果
    models = []

    # 依次训练
    for model_untrained in models_untrained:
        model_name = type(model_untrained).__name__
        print(f"【{model_name}】")
        model, metrics_train, metrics_test = model_train(model_untrained, X_train, y_train, X_test=X_test, y_test=y_test, model_type=model_type)
        models.append(model)
        # 得到测试集评估结果
        eval_dic[model_name] = metrics_test.copy()
        # 得到训练集评估结果
        eval_dic_train[model_name] = metrics_train.copy()
        print("Evaluation on Training Data:")
        print(metrics_train)
        print("Evaluation on Test Data:")
        print(metrics_test)
        # save model
        s = pickle.dumps(model)
        if not os.path.exists(config.model_folder): os.makedirs(config.model_folder)
        with open(config.model_folder + model_name + '.pkl','wb+') as f:#注意此处mode是'wb+'，表示二进制写入
            f.write(s)
        print('\n')
		
    eval_df = plot_evaluation_comparison(eval_dic)

    # find the best model
    metric = 'F1-Score'
    model_perf = eval_df[eval_df['Metric']==metric].sort_values(by='Value',ascending=False)
    best_score = model_perf.iloc[0]['Value']
    best_model_name = model_perf.iloc[0]['Model']
    print(f"Best model is {best_model_name}, {metric} is {best_score:.4f}")

    for model in models:
        if type(model).__name__ == best_model_name:
            best_model = model
            print(f"Best model: {best_model}")
            break



def PCA_n(X_scaled):#主成分分析
	estimator_pca = PCA(n_components=None)
	estimator_pca.fit(X_scaled)
	evr = estimator_pca.explained_variance_ratio_ 
	plt.figure(figsize=(8, 5))
	plt.plot(np.arange(1, len(evr) + 1), np.cumsum(evr*100), "-o")
	plt.title("Cumulative Explained Variance Ratio", fontsize=15)
	plt.xlabel("number of components",fontsize=15)
	plt.ylabel("explained variance ratio(%)",fontsize=15)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.show()

def pcaX(X_scaled,n):
	pca = PCA(n_components=n)
	pca_X = pca.fit_transform(X_scaled)
	print('Decomposition: ',X_scaled.shape,'-->',pca_X.shape)
	return pd.DataFrame(pca_X),pca



# optimization method
def bayes_opt(Model,params:dict, X, y, n_iter = 5, scoring = 'r2', cv = 3):
    """
    BayesSearch One Model
	
    params
    - Model - Model Class
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
    return opt,opt.best_params_,opt.best_score_





# 定义模型性能评估方法
def evaluation(model = None, X_test = None, y_test = None, y_pred = None, model_type = None):
	"""
	传入模型，真实值y_test和预测值y_pred，评估模型效果
	model_type = regression评估回归模型
	model_type = classification评估分类模型
	"""
	if model!=None:
		y_pred = model.predict(X_test)
	# check if onehot encoded:
	if len(y_pred.shape) == 2:
		y_pred = y_pred.argmax(axis=1)
	if len(y_test.shape) == 2:
		y_test = y_test.argmax(axis=1)

	metrics = {}
	# if classification model
	if model_type == 'classification':
		from sklearn.metrics import classification_report
		from sklearn.metrics import accuracy_score   #正确率
		from sklearn.metrics import precision_score	#精准率
		from sklearn.metrics import recall_score	  #召回率
		from sklearn.metrics import f1_score		 #调和平均值F1
		# metrics['cls_report'] = classification_report(y_test, y_pred)
		metrics['accuracy'] = accuracy_score(y_test,y_pred)
		metrics['precision'] =  precision_score(y_test,y_pred,average='macro')
		metrics['recall'] = recall_score(y_test,y_pred,average='macro')
		metrics['f1-score'] = f1_score(y_test,y_pred,average='macro')
		
	# if regression model
	elif model_type== 'regression':
		from sklearn.metrics import mean_squared_error
		from sklearn.metrics import mean_absolute_error
		from sklearn.metrics import r2_score
		metrics['MSE'] = mean_squared_error(y_test,y_pred)
		metrics['RMSE'] = mean_squared_error(y_test,y_pred) ** 0.5
		metrics['MAE'] = mean_absolute_error(y_test,y_pred)
		metrics['R2'] = r2_score(y_test,y_pred)
	else:
		raise Exception('model_type should be classification or regression!')
	return metrics



def model_train(model, X_train, y_train, X_test=None, y_test=None, model_type=None):
    time_start = time.time()
    model.fit(X_train, y_train)
    time_end = time.time()
    time_cost = time_end - time_start
    print(f"- training {type(model).__name__}, time cost: {time_cost:.6f} seconds")
    metrics_test = None
    metrics_train = None
    if X_test is not None:
        y_pred_test = model.predict(X_test)
        metrics_test = evaluation(X_test=X_test, y_test=y_test, y_pred = y_pred_test, model_type=model_type)
        y_pred_train = model.predict(X_train)
        metrics_train = evaluation(X_test=X_test, y_test=y_train, y_pred= y_pred_train, model_type=model_type)
    return model, metrics_train, metrics_test



def plot_evaluation_comparison(eval_dic,metric_x='Metric',metric_hue='Model',dataset_name=None):
	"""
	metric_dic looks like: {'model_name':{'metric_name': value}}
	"""
	# converts to pd.dataframe
	eval_df = pd.DataFrame([[md,mt.title(),v] for md,dic in eval_dic.items() for mt,v in dic.items()],
							columns= ['Model','Metric','Value'])
	eval_df.sort_values(by=['Metric','Value'],inplace=True)
	eval_df.reset_index(drop=True,inplace=True)
	print(eval_df)
	plt.figure(figsize=(10,7))
	sns.barplot(data=eval_df,x = metric_x,y = 'Value',hue=metric_hue)
	plt.title(f"Model Comparison",fontsize=15)
	plt.xticks(rotation=0)
	plt.ylim(eval_df['Value'].min()*0.8,eval_df['Value'].max()*1.05)
	plt.legend(loc = 0, prop = {'size':8})
	plt.tight_layout()
	plt.savefig((f"{dataset_name} - " if dataset_name else '') + f"Comparison.jpg",dpi=300)
	plt.show()
	return eval_df



if __name__ == "__main__":
    main()