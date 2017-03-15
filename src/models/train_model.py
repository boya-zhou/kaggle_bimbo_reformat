
# coding: utf-8

# In[47]:

import pandas as pd
import numpy as np
import xgboost as xgb
import sys

def train_predict(path):
    # In[48]:

    # path = "/home/siyuan/Documents/Boya/bimbo_kaggle_reformat-master"


    # In[49]:

    # hold_data is for for parameter tuining, already got them
    train_data = pd.read_csv(path + "/data/processed/train.csv")
    train_label = pd.read_csv(path + "/data/processed/train_label.csv")
    test_data = pd.read_csv(path + "/data/processed/test_label.csv")


    # In[50]:

    test_data.columns = train_data.columns


    # In[51]:

    # print train_data.shape, train_label.shape


    # In[52]:

    param = {'booster':'gbtree',
             'nthread': 10,
             'max_depth':5,
             'eta':0.2,
             'silent':1,
             'subsample':0.7,
             'objective':'reg:linear',
             'eval_metric':'rmse',
             'colsample_bytree':0.7}


    # In[53]:

    num_round = 566
    dtrain = xgb.DMatrix(train_data, label = train_label, missing= np.nan)
    bst_bimbo = xgb.train(param, dtrain, num_round)
    print 'training finished!'

    bst_bimbo.save_model(path + '/models/xgb_bimbo.model')

    dtest = xgb.DMatrix(test_data, missing= np.nan)
    submission_bimbo = bst_bimbo.predict(dtest)
    pd.DataFrame(submission_bimbo).to_csv("sample_sub.csv", index = False)
    print 'predicting finished!'


# In[ ]:
if __name__ == "__main__":
    train_predict(sys.argv[1])
