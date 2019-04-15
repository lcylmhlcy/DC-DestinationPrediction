# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from math import radians, atan, tan, sin, acos, cos
from sklearn.cluster import DBSCAN, Birch
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

root = '/home/computer/lcy/kaggle/Vechile_End/bayes/'

train = pd.read_csv('train_new.csv', low_memory=False)
test = pd.read_csv('test_new.csv', low_memory=False)
print('train_csv row:  ' + str(train.shape[0])) # 1495814
print('test_csv row:  ' + str(test.shape[0]))  # 58097

# ---------------------------------Prepare lat_lon data-------------------------------------------------------
trL = train.shape[0] * 2
X = np.concatenate([train[['start_lat', 'start_lon']].values,
                    train[['end_lat', 'end_lon']].values,
                    test[['start_lat', 'start_lon']].values])

# ---------------------------Use DBSCN for lat_lon data------------------------------------------------- 
# db = DBSCAN(eps=5e-4, min_samples=3, p=1, leaf_size=10, n_jobs=-1).fit(X)
db = DBSCAN(eps=0.00035, min_samples=3, p=1, leaf_size=30, n_jobs=-1).fit(X)
# db = Birch(n_clusters = None).fit_predict(X)
labels = db.labels_
print('Estimated number of clusters of DBSCN: %d' % len(set(labels)))  # 110227

# ------------------------------ DBSCAN label of train and test---------------------------------------------------------
info = pd.DataFrame(X[:trL,:], columns=['lat', 'lon'])
info['block_id'] = labels[:trL] # DBSCAN label of train
print('The number of miss start block in train data', (info.block_id.iloc[:trL//2] == -1).sum())  # 271285
print('The number of miss end block in train data', (info.block_id.iloc[trL//2:] == -1).sum())  # 274566

test_info = pd.DataFrame(X[trL:,:], columns=['lat', 'lon'])
test_info['block_id'] = labels[trL:] # DBSCAN label of test
print('The number of miss start block in test data', (test_info.block_id == -1).sum())  # 11519

# ----------------------------Concat DBSCAN label to train and test data---------------
train['start_block'] = info.block_id.iloc[:trL//2].values
train['end_block'] = info.block_id.iloc[trL//2:].values
test['start_block'] = test_info.block_id.values
good_train_idx = (train.start_block != -1) & (train.end_block != -1)
print('The number of good training data', good_train_idx.sum())  # 1033722
good_train = train.loc[good_train_idx, :]
print('saving new train & test data')

good_train.to_csv(root+'good_train.csv', index=None)
test.to_csv(root+'good_test.csv', index=None)

# ----------------------------------Generate is_holiday and hour for train and test data--------------------------------------
def transformer(df):
    special_holiday = ['2018-01-01'] + ['2018-02-%d' % d for d in range(15, 22)] + \
                      ['2018-04-%2d' % d for d in range(5, 8)] + \
                      ['2018-04-%d' % d for d in range(29, 31)] + ['2018-05-01'] +\
                      ['2018-06-%d' % d for d in range(16, 19)] + \
                      ['2018-09-%d' % d for d in range(22, 25)] + \
                      ['2018-10-%2d' % d for d in range(1, 8)]
                      
    special_workday = ['2018-02-%d' % d for d in [11, 24]] + \
                      ['2018-04-08'] + ['2018-04-28'] + \
                      ['2018-09-%d' % d for d in range(29, 31)]
                      
    for t_col in ['start_time']:
        tmp = df[t_col].map(pd.Timestamp) # convert to ****-**-** format
        df['hour'] = tmp.map(lambda t: t.hour // 3)  # 3 divide 24 hours into 8 classes
        df['half'] = tmp.map(lambda t: t.minute // 30)  # 30 divide 60 minutes into 2 classes
        df['day'] = tmp.map(lambda t: t.dayofweek)  # divide days into 7 classes (week)
        tmp_date = df[t_col].map(lambda s: s.split(' ')[0])
        not_spworkday_idx = ~tmp_date.isin(special_workday)
        spholiday_idx = tmp_date.isin(special_holiday)
        weekend_idx = (df['day'] >= 5)
        df['is_holiday'] = ((weekend_idx & not_spworkday_idx) | spholiday_idx).astype(int)  # divide into 2 classes (holiday or not)

train = pd.read_csv(root+'good_train.csv', low_memory=False)
test = pd.read_csv(root+'good_test.csv', low_memory=False)
transformer(train)
transformer(test)
train.to_csv(root+'good_train_holiday.csv', index=None)
test.to_csv(root+'good_test_holiday.csv', index=None)

train = pd.read_csv(root+'good_train_holiday.csv', low_memory=False)
test = pd.read_csv(root+'good_test_holiday.csv', low_memory=False)

# -------------------------Calculating conditional probability on train data-------------------------
Probability = {}

## (1)Calculate P(start_block|end_block)
name = 'start_block'
pname = 'P(start_block|end_block)'
print('calculating %s' % pname)
tmp_func = lambda g: (1.0 * g[name].value_counts()) / (len(g) + 10)
tmp = train.groupby('end_block').apply(tmp_func).reset_index() # convert groupby result to DataFrame
tmp.columns = ['end_block', name, pname]
print(tmp.head())
Probability[pname] = tmp
print('\n')

## (2)Calculate P(out_id|end_block)
name = 'out_id'
pname = 'P(out_id|end_block)'
print('calculating %s' % pname)
tmp_func = lambda g: (1.0 * g[name].value_counts()) / (len(g) + 10)
tmp = train.groupby('end_block').apply(tmp_func).reset_index()
tmp.columns = ['end_block', name, pname]
print(tmp.head())
Probability[pname] = tmp
print('\n')

## (3)Calculate P(is_holiday|end_block)
name = 'is_holiday'
pname = 'P(is_holiday|end_block)'
print('calculating %s' % pname)
tmp_func = lambda g: (1.0 * g[name].value_counts() + 3.) / (len(g) + 10)
tmp = train.groupby('end_block').apply(tmp_func).reset_index()
tmp.columns = ['end_block', name, pname]
print(tmp.head())
Probability[pname] = tmp
print('\n')

## (4)Calculate P((is_holiday, hour)|end_block)
pname = 'P((is_holiday, hour)|end_block)'
print('calculating %s' % pname)
tmp_func = lambda g: (5 + 1.0 * g.groupby(['is_holiday', 'hour']).size()) / (len(g))
tmp = train.groupby('end_block').apply(tmp_func).reset_index().rename(columns={0: pname})
print(tmp.head())
Probability[pname] = tmp
print('\n')

## (5)Calculate P(day|end_block)
name = 'day'
pname = 'P(day|end_block)'
print('calculating %s' % pname)
tmp_func = lambda g: 1.0 * g[name].value_counts() / len(g)
tmp = train.groupby('end_block').apply(tmp_func).reset_index()
tmp.columns = ['end_block', name, pname]
print(tmp.head())
Probability[pname] = tmp
print('\n')

## (6)Calculate P(hour|end_block)
name = 'hour'
pname = 'P(hour|end_block)'
print('calculating %s' % pname)
tmp_func = lambda g: 1.0 * g[name].value_counts() / len(g)
tmp = train.groupby('end_block').apply(tmp_func).reset_index()
tmp.columns = ['end_block', name, pname]
print(tmp.head())
Probability[pname] = tmp
print('\n')

## (7)Calculate P((hour,half)|end_block)
pname = 'P((hour,half)|end_block)'
print('calculating %s' % pname)
tmp_func = lambda g: 1.0 * g.groupby(['hour', 'half']).size() / len(g)
tmp = train.groupby('end_block').apply(tmp_func).reset_index().rename(columns={0: pname})
print(tmp.head())
Probability[pname] = tmp
print('\n')

# ----------------------------------Calculate prior probability for train data--------------------------------------------------
pname = 'P(end_block)'
print('calculating %s' % pname)
tmp = train.end_block.value_counts().reset_index()
tmp.columns = ['end_block', pname]
print(tmp.head())
Probability[pname] = tmp
print('\n')

# ----------------------------------Calculate posterior probability of Bayes--------------------------------------------------------- 
#----- P(end_block|(start_block, out_id, is_holiday, hour)) = 
#----- P(end_block) * P(start_block|end_block) * P(out_id|end_block) * P((is_holiday, hour)|end_block)
predict_info = test.copy()

## P(out_id|end_block)
predict_info = predict_info.merge(Probability['P(out_id|end_block)'], on=['out_id'], how='left')
predict_info['P(out_id|end_block)'] = predict_info['P(out_id|end_block)'].fillna(1e-5)

## P(is_holiday|end_block)
predict_info = predict_info.merge(Probability['P(is_holiday|end_block)'], on=['is_holiday', 'end_block'], how='left')
predict_info['P(is_holiday|end_block)'] = predict_info['P(is_holiday|end_block)'].fillna(1e-4)

## P(day|end_block)
predict_info = predict_info.merge(Probability['P(day|end_block)'], on=['day', 'end_block'], how='left')
predict_info['P(day|end_block)'] = predict_info['P(day|end_block)'].fillna(1e-4)

## P((is_holiday, hour)|end_block)
predict_info = predict_info.merge(Probability['P((is_holiday, hour)|end_block)'], on=['is_holiday', 'hour', 'end_block'], how='left')
predict_info['P((is_holiday, hour)|end_block)'] = predict_info['P((is_holiday, hour)|end_block)'].fillna(1e-4)

## P(start_block|end_block)
predict_info = predict_info.merge(Probability['P(start_block|end_block)'], on=['start_block', 'end_block'], how='left')
predict_info['P(start_block|end_block)'] = predict_info['P(start_block|end_block)'].fillna(1e-5)

## P(end_block)
predict_info = predict_info.merge(Probability['P(end_block)'], on='end_block', how='left')
predict_info['P(end_block)'] = predict_info['P(end_block)'].fillna(1e-1)

predict_info['P(end_block|(start_block, out_id, is_holiday, hour))'] = predict_info['P((is_holiday, hour)|end_block)'] * \
                                                    predict_info['P(out_id|end_block)'] * \
                                                    predict_info['P(start_block|end_block)'] * \
                                                    predict_info['P(end_block)']


# ----------------------------------------generate DBSCAN label-----------------------------------------------------------------------------
block_lat_lon = train.groupby('end_block')[['end_lat', 'end_lon']].mean().reset_index()
predict_info = predict_info.merge(block_lat_lon, on='end_block', how='left')

predict_info.to_csv(root+'predict_info.csv', index=None)

which_probability = 'P(end_block|(start_block, out_id, is_holiday, hour))'
predict_info = pd.read_csv(root+'predict_info.csv', low_memory=False)
test = pd.read_csv(root+'good_test_holiday.csv', low_memory=False)

print(predict_info[['start_lat', 'start_lon', 'end_lat', 'end_lon']].describe())

predict_result = predict_info.groupby('r_key').apply(lambda g: g.loc[g[which_probability].idxmax(), :]).reset_index(drop=True)

output_result = test[['r_key', 'start_lat', 'start_lon']].merge(predict_result[['r_key', 'end_lat', 'end_lon']], on='r_key', how='left')
# print(output_result.end_lat.isnull().sum())

nan_idx = output_result.end_lat.isnull()
output_result.loc[nan_idx, 'end_lat'] = output_result['start_lat'][nan_idx]
output_result.loc[nan_idx, 'end_lon'] = output_result['start_lon'][nan_idx]
#output_result[['start_lat', 'end_lat', 'end_lon']].describe()
print(output_result.head())
# print(output_result.info())

output_result[['r_key', 'end_lat', 'end_lon']].to_csv(root+'bayes.csv', index=None)