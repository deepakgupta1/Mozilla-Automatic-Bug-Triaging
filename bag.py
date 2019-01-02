import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

X1 = pd.read_csv('10folds_10l', delimiter = ',', header = None)
X2 = pd.read_csv('attention', delimiter = ',', header = None)
X3 = pd.read_csv('attention_glove', delimiter = ',', header = None)
X4 = pd.read_csv('attention_subword', delimiter = ',', header = None)
X5 = pd.read_csv('attention_wiki_small', delimiter = ',', header = None)

print(X1.shape, X2.shape, X3.shape, X4.shape, X5.shape)

dataframe = pd.read_csv('train_test_total_bug_desc.csv')
print('dataframe read shape: ' + str(dataframe.shape))

df = dataframe.groupby('component_id')['bug_id'].count().reset_index().sort_values('bug_id', ascending=False)
more_than_one = df[df['bug_id'] > 1]['component_id'].tolist()
#more_than_one = df['component_id'][:1000].tolist()
print(len(more_than_one))
dataframe = dataframe[dataframe['component_id'].isin(more_than_one)]
print('dataframe shape: ' + str(dataframe.shape))

class_le = LabelEncoder()
dataframe['component_id'] = class_le.fit_transform(dataframe['component_id'])

for c in X1.columns:
	X1[c] = 0.95*((X1[c] + X2[c] + X3[c])/3.0) + 0.05*((X4[c] + X5[c])/2.0)
	
predict = X1.values

component_id = np.argmax(predict, axis=1)
component_id = class_le.inverse_transform(component_id)
confidence = np.max(predict, axis=1)

submit = pd.DataFrame({'component_id':component_id, 'confidence':confidence})
submit.to_csv('submit_mean_all_five.csv', index=False)

#np.savetxt('attention', predict, delimiter=',', fmt = '%0.6f')