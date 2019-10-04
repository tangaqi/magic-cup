# -*- coding: utf-8 -*-
"""
@file:base_6900.py
@time:2019/7/6 21:49
@author:Tangj
@software:Pycharm
@Desc
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from scipy.stats import kurtosis
import time
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
'''
    方案思路：33分类。
              label为规定还款日期距还款日期的天数，可能的情况有0天到31天，未还款定义为-1，一共33个类别。
              预测出每个label对应的概率，然后分别乘以应还的金额，就是每天需要还的金额。
              这里可以将预测概率非常小的还款直接归0，因为实际情况来说，只有一天还款，且还款金额是全部借款数。
'''
# 读取数据，并将相应的日期解析为时间格式
train_data = pd.read_csv('../data/train.csv', parse_dates=['auditing_date', 'due_date', 'repay_date'])
train_data['repay_date'] = train_data[['due_date', 'repay_date']].apply(
    lambda x: x['repay_date'] if x['repay_date'] != '\\N' else x['due_date'], axis=1
)
train_data['repay_amt'] = train_data['repay_amt'].apply(lambda x: x if x != '\\N' else 0).astype('float32')
train_data['label'] = (train_data['due_date'] - train_data['repay_date']).dt.days
train_data.loc[train_data['repay_amt'] == 0, 'label'] = -1

# 读取user_repay_logs文件扩充训练集，order_id是为了区分原始的train和扩充的train
train_data.loc[:, 'order_id'] = 0
train_expand = pd.read_csv('../data/user_repay_logs.csv', parse_dates=['due_date', 'repay_date'])
train_expand = train_expand[train_expand['order_id'] == 1]
# del train_expand['order_id']
train_expand.loc[train_expand['repay_date'].dt.year == 2200, 'repay_amt'] = 0
train_expand['label'] = (train_expand['due_date'] - train_expand['repay_date']).dt.days
train_expand.loc[train_expand['repay_amt'] == 0, 'label'] = -1
train_data = pd.concat([train_data, train_expand])
train_data = train_data.drop_duplicates('listing_id').reset_index(drop=True)

# 扩充的训练集中有的不是2018年的数据，因此这里只保留2018年的数据作为train集合
mask = train_data['due_date'].dt.year == 2018
train_data = train_data[mask]
clf_labels = train_data['label'].values + 1
amt_labels = train_data['repay_amt'].values
del train_data['label'], train_data['repay_amt'], train_data['repay_date']
train_due_amt_data = train_data[['due_amt']]
train_num = train_data.shape[0]

# 对test集合的处理
test_data = pd.read_csv('../data/test.csv', parse_dates=['auditing_date', 'due_date'])
sub = test_data[['listing_id', 'auditing_date', 'due_amt', 'due_date']]
data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

listing_info_data = pd.read_csv('../data/listing_info.csv')
del listing_info_data['user_id'], listing_info_data['auditing_date']
data = data.merge(listing_info_data, on='listing_id', how='left')

# 将user信息加入进来，表中有少数user不止一条记录，因此按日期排序，去重，只保留最新的一条记录。
user_info_data = pd.read_csv('../data/user_info.csv', parse_dates=['reg_mon', 'insertdate'])
user_info_data.rename(columns={'insertdate': 'info_insert_date'}, inplace=True)
user_info_data_1 = user_info_data.sort_values(by='info_insert_date', ascending=False).drop_duplicates('user_id').reset_index(drop=True)#按照插入日期降序排列，去重，只保留最新的一条
user_info_data_1['foreign_land']=np.where(user_info_data_1['cell_province']==user_info_data_1['id_province'],'n','y')
modifyInfoNum=user_info_data.groupby('user_id').count()['info_insert_date'].to_frame().rename(columns={'info_insert_date':'modify_info_num'})
user_info_data_2=pd.merge(user_info_data_1,modifyInfoNum,how='left',on='user_id')

# 将user信息中的年龄信息分桶
def map_age(s):
    if s < 25:
        return 'Young'
    elif s>24 and s < 36:
        return 'Middle1'
    elif s>35 and s < 51:
        return 'Middle2'
    else:
        return 'Old'
user_info_data_2['map_age']=user_info_data_2['age'].map(map_age)
data = data.merge(user_info_data_2, on='user_id', how='left')#将用户基础信息表合并到训练集之中

# 将用户画像标签列表信息加入，对于多于的数据，排序去重合并
user_tag_data = pd.read_csv('../data/user_taglist.csv', parse_dates=['insertdate'])
user_tag_data.rename(columns={'insertdate': 'tag_insert_date'}, inplace=True)
user_tag_data_1 = user_tag_data.sort_values(by='tag_insert_date', ascending=False).drop_duplicates('user_id').reset_index(drop=True)
modifyTagListNum = user_tag_data.groupby('user_id').count()['tag_insert_date'].to_frame().rename(columns={'tag_insert_date':'modify_taglist_num'})
user_tag_data_2=pd.merge(user_tag_data_1,modifyTagListNum,how='left',on='user_id')
data = data.merge(user_tag_data_2, on='user_id', how='left')

# 用户行为表
user_behavior_logs = pd.read_csv('../data/user_behavior_logs.csv', parse_dates=['behavior_time'])
user_behavior_logs_1=user_behavior_logs.groupby('user_id').count()['behavior_type'].to_frame().rename(columns={'behavior_type':'behavior_num'})
data = data.merge(user_behavior_logs_1, on='user_id', how='left')
# 基于全部还款记录计算每位user的逾期率
user_repay_logs=pd.read_csv('../data/user_repay_logs.csv',index_col=None)
user_repay_logs['expire']=np.where(user_repay_logs['repay_date']=='2200-01-01',1,0)
expire_cnt_ratio=user_repay_logs.groupby('user_id')['expire'].agg({'repay_mean':'mean'}).reset_index()
data = data.merge(expire_cnt_ratio, on='user_id', how='left')

repay_log_data = pd.read_csv('../data/user_repay_logs.csv', parse_dates=['due_date', 'repay_date'])
# 用户还款日志表
# 由于题目任务只预测第一期的还款情况，因此这里只保留第一期的历史记录。
repay_log_data = repay_log_data[repay_log_data['order_id'] == 1].reset_index(drop=True)
repay_log_data['repay'] = repay_log_data['repay_date'].astype('str').apply(lambda x: 1 if x != '2200-01-01' else 0)
repay_log_data['early_repay_days'] = (repay_log_data['due_date'] - repay_log_data['repay_date']).dt.days
repay_log_data['early_repay_days'] = repay_log_data['early_repay_days'].apply(lambda x: x if x >= 0 else -1)
for f in ['listing_id', 'order_id', 'due_date', 'repay_date', 'repay_amt']:
    del repay_log_data[f]
group = repay_log_data.groupby('user_id', as_index=False)
repay_log_data = repay_log_data.merge(
    group['repay'].agg({'repay_mean': 'mean'}), on='user_id', how='left'
)
repay_log_data = repay_log_data.merge(
    group['early_repay_days'].agg({
        'early_repay_days_max': 'max', 'early_repay_days_median': 'median', 'early_repay_days_sum': 'sum',
        'early_repay_days_mean': 'mean', 'early_repay_days_std': 'std'
    }), on='user_id', how='left'
)
repay_log_data = repay_log_data.merge(
    group['due_amt'].agg({
        'due_amt_max': 'max', 'due_amt_min': 'min', 'due_amt_median': 'median',
        'due_amt_mean': 'mean', 'due_amt_sum': 'sum', 'due_amt_std': 'std',
        'due_amt_skew': 'skew', 'due_amt_kurt': kurtosis, 'due_amt_ptp': np.ptp
    }), on='user_id', how='left'
)
del repay_log_data['repay'], repay_log_data['early_repay_days'], repay_log_data['due_amt']
repay_log_data = repay_log_data.drop_duplicates('user_id').reset_index(drop=True)
data = data.merge(repay_log_data, on='user_id', how='left')

cate_cols = ['gender', 'cell_province', 'id_province', 'id_city','foreign_land','map_age']
for f in cate_cols:
    data[f] = data[f].map(dict(zip(data[f].unique(), range(data[f].nunique())))).astype('int32')

data['due_amt_per_days'] = data['due_amt'] / (train_data['due_date'] - train_data['auditing_date']).dt.days
date_cols = ['auditing_date', 'due_date', 'reg_mon', 'info_insert_date', 'tag_insert_date']
for f in date_cols:
    data[f + '_month'] = data[f].dt.month
    if f in ['auditing_date', 'due_date', 'info_insert_date', 'tag_insert_date']:
        data[f + '_day'] = data[f].dt.day
        data[f + '_dayofweek'] = data[f].dt.dayofweek

# 将交叉特征读入
fea_c = ['listing_id', 'age|rate', 'due_amt|rate', 'principal|rate']
train_cross = pd.read_csv('../src/train_2.csv')[fea_c]
test_cross = pd.read_csv('../src/test_2.csv')[fea_c]
cross_fea = pd.concat([train_cross, test_cross])
data = pd.merge(data, cross_fea, on='listing_id', how='left')
del data['listing_id'], data['taglist']

data['user_id'] = data['user_id'].astype('category')
data['due_amt'] = data['due_amt'].astype('category')
data['cnt_uid'] = data['user_id'].map(data['user_id'].value_counts())

Train_values, test_values = data[:train_num], data[train_num:]

# 用时序验证方法，构造线下验证集
mask = Train_values['due_date'].dt.month == 12
train_values = Train_values[~mask]
val_values = Train_values[mask]
y_train = clf_labels[~mask]
y_val = clf_labels[mask]

print(train_values.shape)
date_cols = ['auditing_date', 'due_date', 'reg_mon', 'info_insert_date', 'tag_insert_date']
Train_values.drop(columns=date_cols, axis=1, inplace=True)
train_values.drop(columns=date_cols, axis=1, inplace=True)
val_values.drop(columns=date_cols, axis=1, inplace=True)
test_values.drop(columns=date_cols, axis=1, inplace=True)

clf = LGBMClassifier(
    learning_rate=0.06,
    n_estimators=500,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8
)
amt_oof = np.zeros(train_num)
prob_oof = np.zeros((train_num, 33))
clf.fit(
        Train_values, clf_labels,
        eval_set=[(train_values, y_train), (val_values, y_val)],
        early_stopping_rounds=100, verbose=5
    )

test_pred_prob = clf.predict_proba(test_values.values, num_iteration=clf.best_iteration_)
prob_cols = ['prob_{}'.format(i) for i in range(33)]#prob_0 至 prob_32
for i, f in enumerate(prob_cols):#遍历每一个prob_i
    sub[f] = test_pred_prob[:, i] 
sub_example = pd.read_csv('../data/submission.csv', parse_dates=['repay_date'])
sub_example = sub_example.merge(sub, on='listing_id', how='left')
sub_example['days'] = (sub_example['due_date'] - sub_example['repay_date']).dt.days
# shape = (-1, 33)
test_prob = sub_example[prob_cols].values
test_labels = sub_example['days'].values
test_prob = [test_prob[i][test_labels[i] + 1] for i in range(test_prob.shape[0])]#第i个样本第
sub_example['repay_amt'] = sub_example['due_amt'] * test_prob #第i个样本第test_labels[i]天的预测的还款金额
sub_example[['listing_id', 'repay_date', 'repay_amt']].to_csv('test_cate_count_sub.csv', index=False)