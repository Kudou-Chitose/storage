import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import GridSearchCV

dataset1 = pd.read_csv('data/dataset1.csv')
dataset1.label.replace(-1,0,inplace=True)
dataset2 = pd.read_csv('data/dataset2.csv')
dataset2.label.replace(-1,0,inplace=True)
dataset3 = pd.read_csv('data/dataset3.csv')

dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset3.drop_duplicates(inplace=True)

dataset12 = pd.concat([dataset1,dataset2],axis=0)

dataset1_y = dataset1.label
dataset1_x = dataset1.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)  # 'day_gap_before','day_gap_after' cause overfitting, 0.77
dataset2_y = dataset2.label
dataset2_x = dataset2.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)
dataset12_y = dataset12.label
dataset12_x = dataset12.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)
dataset3_preds = dataset3[['user_id','coupon_id','date_received']]
dataset3_x = dataset3.drop(['user_id','coupon_id','date_received','day_gap_before','day_gap_after'],axis=1)



param_test = {
    'max_depth': range(5,15,2),
    'num_leaves': range(10,40,5),
}

estimator = LGBMRegressor(
    num_leaves = 50,
    max_depth = 13,
    learning_rate =0.1, 
    n_estimators = 1000, 
    objective = 'regression', 
    min_child_weight = 1, 
    subsample = 0.8,
    colsample_bytree=0.8,
    nthread = 7,
)
gsearch = GridSearchCV( estimator , param_grid = param_test, scoring='roc_auc', cv=5)
gsearch.fit(dataset1_x, dataset1_y)
gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_
print_best_score(gsearch,param_test)