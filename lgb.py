import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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

lgb_train, lgb_val = train_test_split(dataset12, test_size=0.3, random_state=0)

lgb_train_y = lgb_train.label
lgb_train_x = lgb_train.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)
lgb_val_y = lgb_val.label
lgb_val_x = lgb_val.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)

lgb_train = lgb.Dataset(lgb_train_x, lgb_train_y)
lgb_val = lgb.Dataset(lgb_val_x, lgb_val_y)


seed = 13
params = {
    'config':'',
    'application':'regression',
    'num_iterations': 2500,
    'learning_rate': 0.05,
    'tree_learner': 'serial',
    'min_data_in_leaf': 50,
    'metric':'auc',
    'feature_fraction': 0.7,
    'feature_fraction_seed': seed,
    'bagging_fraction':1,
    'bagging_freq':10,
    'bagging_seed':seed,
    'metric_freq':1,
    'early_stopping_round':100,
}

model = lgb.train(params,lgb_train,valid_sets=lgb_val)

dataset3_preds['label'] = model.predict(dataset3_x)

dataset3_preds.label = MinMaxScaler().fit_transform(dataset3_preds.label.reshape(-1, 1))

dataset3_preds.sort_values(by=['coupon_id','label'],inplace=True)

dataset3_preds.to_csv("lgb_preds.csv",index=None,header=None)

print dataset3_preds.describe()