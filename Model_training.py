from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold 
from sklearn.neighbors import KNeighborsRegressor
from Hyperparameter.Search_SVR import GridSearchCV_SVR, RandomizedSearchCV_SVR, BayesSearchCV_SVR
from tqdm import tqdm 
from joblib import dump
import time
import seaborn as sns
import os
start_time = time.time()

filepath = r''
resultpath =  r''
data = pd.read_excel(filepath, sheet_name='Sheet')  

Feature_data = data.iloc[:, :].values
MPS_data =  data.iloc[:, :].values
Features_data = np.array(Feature_data)
MPS_data = np.array(MPS_data)
X = Features_data  
Y = MPS_data


######################################################################
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)) 
X_minmax = min_max_scaler.fit_transform(X)
X = X_minmax
min_val = np.min(Y)
max_val = np.max(Y)
normalized_vector = (Y - min_val) / (max_val - min_val)
Y = normalized_vector  
######################################################################

######################################################################
PCC_data = np.array(np.column_stack((Feature_data, MPS_data)), dtype=float)
pcc_matrix = np.corrcoef(PCC_data.T)
mask = np.triu(np.ones_like(pcc_matrix, dtype=bool), k=1)
plt.rc('font',family='Times New Roman')
plt.rcParams.update({'font.size': 25})
fig, ax = plt.subplots(figsize=(18, 18))

heatmap = sns.heatmap(
    pcc_matrix,
    annot=True,
    fmt=".2f",
    vmax=1,
    vmin=-1,
    xticklabels=True,
    yticklabels=True,
    square=True,
    cmap="RdBu_r",
    mask=mask,  
    annot_kws={"size": 15} ,
    cbar_kws={
        "shrink": 0.8, 
        "aspect": 30, 
        "pad": 0.02   
    }
)
######################################################################

######################################################################
with open(fr"{resultpath}\HyperParameter_optimization\optimization", "w") as file:
    best_model_grid, best_params_grid, best_score_grid = GridSearchCV_SVR(X, Y)             
    best_model_grid.fit(X, Y)
    Y_pre_grid = best_model_grid.predict(X)
    file.write("网格搜索 SVR\n")
    file.write(f"最优参数: {best_params_grid}\n")
    file.write(f"最优分数: {best_score_grid}\n")
    file.write(f"决定系数 R2: {r2_score(Y, Y_pre_grid)}\n")
    file.write(f"均方根误差 RMSE: {np.sqrt(mean_squared_error(Y, Y_pre_grid))}\n\n")
    print("网格搜索决定系数 R2:", r2_score(Y, Y_pre_grid))
    print("网格搜索均方根误差RMSE:", np.sqrt(mean_squared_error(Y, Y_pre_grid)))
    print()
    plt.figure()
    plt.scatter(Y, Y_pre_grid, color='blue')
    plt.xlim(0.6,2.0)
    plt.ylim(0.6,2.0)
    plt.gca().set_aspect('equal')
    plt.title('GridSearchCV_SVR')
    plt.xlabel("True value")
    plt.ylabel("Predict value")
    
    best_model_random, best_params_random, best_score_random = RandomizedSearchCV_SVR(X, Y)    
    best_model_random.fit(X, Y)
    Y_pre_random = best_model_random.predict(X)
    file.write("随机搜索 SVR\n")
    file.write(f"最优参数: {best_params_random}\n")
    file.write(f"最优分数: {best_score_random}\n")
    file.write(f"决定系数 R2: {r2_score(Y, Y_pre_random)}\n")
    file.write(f"均方根误差 RMSE: {np.sqrt(mean_squared_error(Y, Y_pre_random))}\n\n")
    print("随机搜索决定系数 R2:", r2_score(Y, Y_pre_random))
    print("随机搜索均方根误差RMSE:", np.sqrt(mean_squared_error(Y, Y_pre_random)))
    print()
    plt.figure()
    plt.scatter(Y, Y_pre_random, color='blue')
    plt.xlim(0.6,2.0)
    plt.ylim(0.6,2.0)
    plt.gca().set_aspect('equal')
    plt.title('RandomizedSearchCV_SVR')
    plt.xlabel("True value")
    plt.ylabel("Predict value")

    best_model_bayes, best_params_bayes, best_score_bayes = BayesSearchCV_SVR(X, Y)         
    best_model_bayes.fit(X, Y)
    Y_pre_bayes = best_model_bayes.predict(X)
    file.write("贝叶斯优化 SVR\n")
    file.write(f"最优参数: {best_params_bayes}\n")
    file.write(f"最优分数: {best_score_bayes}\n")
    file.write(f"决定系数 R2: {r2_score(Y, Y_pre_bayes)}\n")
    file.write(f"均方根误差 RMSE: {np.sqrt(mean_squared_error(Y, Y_pre_bayes))}\n\n")
    print("贝叶斯优化决定系数 R2:", r2_score(Y, Y_pre_bayes))
    print("贝叶斯优化均方根误差 RMSE:", np.sqrt(mean_squared_error(Y, Y_pre_bayes)))
    print()
    plt.figure()
    plt.scatter(Y, Y_pre_bayes, color='blue')
    plt.xlim(0.6,2.0)
    plt.ylim(0.6,2.0)
    plt.gca().set_aspect('equal')
    plt.title('BayesSearchCV_SVR')
    plt.xlabel("True value")
    plt.ylabel("Predict value")
    plt.show()
    
    dump(best_model_grid, fr'{resultpath}\HyperParameter_optimization\5F_grid_SVR.m')          
    dump(best_model_random, fr'{resultpath}\HyperParameter_optimization\5F_random_SVR.m')
    dump(best_model_bayes, fr'{resultpath}\HyperParameter_optimization\5F_bayes_SVR.m')  
######################################################################

######################################################################
Model_23F = ['23F_grid_SVR','23F_random_SVR','23F_bayes_SVR','23F_grid_RF','23F_bayes_RF',
              '23F_random_XG','23F_bayes_XG','23F_bayes_AdaSVR','23F_bayes_GPR','23F_grid_KNN']  
os.makedirs(fr'{resultpath}\HyperParameter_optimization\picture', exist_ok=True)
os.makedirs(fr'{resultpath}\HyperParameter_optimization\detailed_results', exist_ok=True)

for model_name in Model_23F:
    model_pic_dir = os.path.join(resultpath, 'HyperParameter_optimization', 'picture', model_name)
    os.makedirs(model_pic_dir, exist_ok=True)

results = []
detailed_results_path = fr'{resultpath}\HyperParameter_optimization\detailed_results\all_folds_results.xlsx'
with pd.ExcelWriter(detailed_results_path, engine='openpyxl') as writer:
    for model_name in tqdm(Model_23F, desc="Evaluating models"):
        clf = load(fr'{resultpath}\HyperParameter_optimization\{model_name}.m')
        clf = KNeighborsRegressor(algorithm='ball_tree', n_neighbors=1, weights='uniform')

        model_pic_dir = os.path.join(resultpath, 'HyperParameter_optimization','picture', model_name)
        
        all_train_r2 = []
        all_test_r2 = []
        all_train_rmse = []
        all_test_rmse = []
        model_detailed_results = [] 
        
        def create_stratification_bins(y, n_splits=5):  
            return pd.qcut(y, n_splits, labels=False, duplicates='drop')
        
        for repeat in range(100):
            stratification_bins = create_stratification_bins(Y, n_splits=5)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat)
            repeat_train_r2 = []
            repeat_test_r2 = []
            repeat_train_rmse = []
            repeat_test_rmse = []
            for fold, (train_index, test_index) in enumerate(skf.split(X, stratification_bins)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]
                
                clf.fit(X_train, y_train)
                train_pred = clf.predict(X_train)
                test_pred = clf.predict(X_test)
                
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                
                repeat_train_r2.append(train_r2)
                repeat_test_r2.append(test_r2)
                repeat_train_rmse.append(train_rmse)
                repeat_test_rmse.append(test_rmse)
                
                model_detailed_results.append({
                    'Repeat': repeat + 1,
                    'Fold': fold + 1,
                    'Type': 'Train',
                    'R2': train_r2,
                    'RMSE': train_rmse
                })
                model_detailed_results.append({
                    'Repeat': repeat + 1,
                    'Fold': fold + 1,
                    'Type': 'Test',
                    'R2': test_r2,
                    'RMSE': test_rmse
                })
                
                if repeat < 10:
                    plt.figure(figsize=(6, 6))
                    plt.scatter(y_train, train_pred, color='lightblue', alpha=0.7, 
                               edgecolor='dodgerblue', label=f'Train R² = {train_r2:.3f}')
                    plt.scatter(y_test, test_pred, color='lightcoral', alpha=0.7,
                               edgecolor='red', label=f'Test   R² = {test_r2:.3f}')
                    plt.plot([0.7, 2.1], [0.7, 2.1], '--', color='dimgray', linewidth=1.5)
                    
                    plt.xlim(0.7, 2.1)
                    plt.ylim(0.7, 2.1)
                    ticks = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
                    plt.xticks(ticks)
                    plt.yticks(ticks)
                    plt.gca().set_aspect('equal')
                    plt.xlabel('Actual value/mm', fontsize=24)
                    plt.ylabel('Predicted value/mm', fontsize=24)
                    
                    ax = plt.gca()
                    ax.tick_params(axis='both', which='both', direction='in',
                                  labelsize=22, bottom=True, left=True,
                                  top=False, right=False)
                    
                    plt.legend(loc='upper left', frameon=False, fontsize=15)
                    plt.rcParams['font.family'] = 'Times New Roman'
                    plt.savefig(
                        os.path.join(model_pic_dir, f'repeat_{repeat+1}_fold_{fold+1}.jpg'), 
                        dpi=600, 
                        bbox_inches='tight'
                    )
                    plt.close()
            
            all_train_r2.append(np.mean(repeat_train_r2))
            all_test_r2.append(np.mean(repeat_test_r2))
            all_train_rmse.append(np.mean(repeat_train_rmse))
            all_test_rmse.append(np.mean(repeat_test_rmse))
    
        avg_train_r2 = np.mean(all_train_r2)
        avg_test_r2 = np.mean(all_test_r2)
        avg_train_rmse = np.mean(all_train_rmse)
        avg_test_rmse = np.mean(all_test_rmse)
        
        print(f"\n{model_name}")
        print(f"100次重复五折交叉验证平均训练集 R²: {avg_train_r2:.4f}")
        print(f"100次重复五折交叉验证平均测试集 R²: {avg_test_r2:.4f}")
        print(f"100次重复五折交叉验证平均训练集 RMSE: {avg_train_rmse:.4f}")
        print(f"100次重复五折交叉验证平均测试集 RMSE: {avg_test_rmse:.4f}")
        
        results.append({
            '模型': model_name,
            '训练集 R2': avg_train_r2,
            '测试集 R2': avg_test_r2,
            '训练集 RMSE': avg_train_rmse,
            '测试集 RMSE': avg_test_rmse
        })
        df_model_detailed = pd.DataFrame(model_detailed_results)
        df_model_detailed.to_excel(writer, sheet_name=model_name[:31], index=False)  

df_results = pd.DataFrame(results)
df_results.to_excel(fr'{resultpath}\HyperParameter_optimization\model_5F_evaluation.xlsx', index=False)
print("\n所有模型评估完成！结果已保存。")
print(f"详细结果已保存至: {detailed_results_path}")
######################################################################