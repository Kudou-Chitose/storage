import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

dataset12_x = dataset12_x.fillna({x:-1 for x in dataset12_x}, inplace=True)
print dataset12_x
dataset3_x = dataset3_x.fillna({x:-1 for x in dataset3_x}, inplace=True)

regr = RandomForestRegressor(random_state=0, n_estimators=100)
regr.fit(dataset12_x, dataset12_y)

dataset3_preds['label'] = regr.predict(dataset3_x)

dataset3_preds.label = MinMaxScaler().fit_transform(dataset3_preds.label.reshape(-1, 1))

dataset3_preds.sort_values(by=['coupon_id','label'],inplace=True)

dataset3_preds.to_csv("rf_preds.csv",index=None,header=None)

print dataset3_preds.describe()