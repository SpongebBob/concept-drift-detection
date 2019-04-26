#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn import metrics


# In[2]:


random_state = 42
np.random.seed(random_state)


# In[3]:


def gen_fake_norm_dateset(column_size=20, instance_size=100000):
    """
    Input size: total batch size
    Distribution: gen a fake dataset for test, 20 coloumns is normal distributaion.
    """
    dataset = {}
    for i in range(column_size):
        dataset['col_{}'.format(i)] = np.random.normal(0,1,instance_size)
    df = pd.DataFrame(dataset)
    train = df[:instance_size//2]
    test = df[instance_size//2:]
    # add drift to column 0
    test['col_0'] += np.random.normal(0.1,0.5,len(test))
    return train, test


# In[4]:


batch1, batch2 = gen_fake_norm_dateset()


# In[5]:


def train_test_split(X, y, test_size, random_state=2018):
    """
    split data to train and test
    """
    sss = list(StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state).split(X, y))
    X_train = np.take(X, sss[0][0], axis=0)
    X_test = np.take(X, sss[0][1], axis=0)
    y_train = np.take(y, sss[0][0], axis=0)
    y_test = np.take(y, sss[0][1], axis=0)
    return [X_train, X_test, y_train, y_test]


# In[13]:


def get_fea_importance(clf, feature_name):
    """
    get feature importance from lightGBM
    """
    gain = clf.feature_importance('gain')
    importance_df = pd.DataFrame({
        'feature':clf.feature_name(),
        'split': clf.feature_importance('split'),
        'gain': gain, # * gain / gain.sum(),
        'gain_percent':100 *gain / gain.sum(),
        }).sort_values('gain',ascending=False)
    return importance_df


# In[14]:


def adversial_validation(batch1, batch2):
    """
    split two batch to get importance
    """
    feature_name = list(batch1.columns)
    train_X = batch1
    train_Y = np.ones(train_X.shape[0])
    test_X = batch2
    test_Y = np.zeros(test_X.shape[0])
    X = np.concatenate((train_X.values,test_X.values),axis=0)
    y = np.concatenate((train_Y,test_Y),axis=0)
    test_size = int(len(X)/5) 
    X, X_test, y, y_test = train_test_split(X, y, test_size, random_state = 42)
    para = {
        'num_leaves': 6,
        'learning_rate': 0.1,
        'bagging_fraction': 0.2, 
        'feature_fraction': 0.5,
        'max_depth': 3, 
        "objective": "binary", 
        "metric":"auc", 
        'verbose': -1, 
        "seed": 42, 
        'num_threads': 8,
    }
    lgb_train = lgb.Dataset(X, y, free_raw_data=True)
    lgb_val = lgb.Dataset(X_test, y_test, free_raw_data=True, reference=lgb_train)
    lgb_model = lgb.train(para, lgb_train, valid_sets=lgb_val, valid_names='eval',feature_name=feature_name,
                                verbose_eval=False, early_stopping_rounds=10, num_boost_round=50)
    fpr, tpr, thresholds = metrics.roc_curve(
        y_test, lgb_model.predict(X_test, num_iteration = lgb_model.best_iteration))
    auc = metrics.auc(fpr, tpr)
    print("----Adversial Score is {}------".format(auc))
    fea_importance_adversial = get_fea_importance(lgb_model, feature_name)
    print(fea_importance_adversial.head(10))
    return fea_importance_adversial, auc


# ### get the batch split result, feature importance and auc

# In[15]:


fea_imp, auc_true = adversial_validation(batch1, batch2)


# ### Estimate the threshold. We could run more to get a distribution 

# In[17]:


estimate_thres_auc = []
estimate_thres_gain = []
for i in range(5):
    len_batch1 = len(batch1) 
    base_df = batch1.append(batch2).reset_index(drop = False).sample(frac=1)
    fea_base, auc_base = adversial_validation(base_df[:len_batch1], base_df[len_batch1:])
    estimate_thres_auc.append(auc_base)
    estimate_thres_gain.append(fea_base['gain'].values[0])


# In[18]:


#auc threashold
np.mean(estimate_thres_auc)


# In[19]:


# drift threashold
np.mean(estimate_thres_gain)

