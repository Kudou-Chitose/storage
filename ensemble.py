import pandas as pd

dataset1 = pd.read_csv('xgb_preds.csv',header=None)
dataset1.columns = ['user_id','coupon_id','date_received','probability1']
dataset2 = pd.read_csv('lgb_preds.csv',header=None)
dataset2.columns = ['user_id','coupon_id','date_received','probability2']
dataset3 = pd.read_csv('rf_preds.csv',header=None)
dataset3.columns = ['user_id','coupon_id','date_received','probability3']

dataset1.sort_values(by=['user_id','coupon_id','date_received'],inplace=True)
dataset2.sort_values(by=['user_id','coupon_id','date_received'],inplace=True)
dataset3.sort_values(by=['user_id','coupon_id','date_received'],inplace=True)

dataset4_preds = pd.merge(dataset1, dataset2, on=['user_id','coupon_id','date_received'])
dataset4_preds = pd.merge(dataset4_preds, dataset3, on=['user_id','coupon_id','date_received'])
dataset4_preds.eval('probability=0.43*probability1+0.57*probability2+0.0*probability3', inplace=True)

dataset4_preds.drop(['probability1','probability2','probability3'],axis=1,inplace=True)
dataset4_preds.sort_values(by=['coupon_id','probability'],inplace=True)
dataset4_preds.to_csv("final_preds.csv",index=None,header=None)