import numpy as np
import pandas as pd

train = pd.read_csv('bugs-training.csv')
test = pd.read_csv('bugs-testing.csv')
longdesc = pd.read_csv('longdescs.csv')

test['component_id'] = 111
train.drop(['assigned_to', 'version'], axis=1, inplace=True)
train_test = pd.concat([train, test])

train_test = train_test.merge(longdesc, on='bug_id', how='left')

train_test['bug_long_desc'] = train_test['long_desc']
train_test.drop('long_desc', axis=1, inplace=True)

train_test['bug_long_desc'].fillna('none', inplace=True)

train_test['total_bug_desc'] = train_test['short_desc'] + '. ' + train_test['bug_long_desc']

train_test[['bug_id', 'total_bug_desc', 'component_id']].to_csv('train_test_total_bug_desc.csv', index=False)