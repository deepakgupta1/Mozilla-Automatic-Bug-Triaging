import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import MaxPooling1D, CuDNNGRU, Bidirectional, Dense, Dropout, Activation, Input, Embedding, Flatten, LSTM, SpatialDropout1D, Conv1D, GlobalMaxPool1D, Concatenate
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

dataframe = pd.read_csv('train_test_total_bug_desc.csv')
print('dataframe read shape: ' + str(dataframe.shape))

df = dataframe.groupby('component_id')['bug_id'].count().reset_index().sort_values('bug_id', ascending=False)
more_than_one = df[df['bug_id'] > 1]['component_id'].tolist()
#more_than_one = df['component_id'][:1000].tolist()
print(len(more_than_one))
dataframe = dataframe[dataframe['component_id'].isin(more_than_one)]
print('dataframe shape: ' + str(dataframe.shape))

MAX_LENGTH = 300
tokenizer = Tokenizer(num_words=1000000)
tokenizer.fit_on_texts(dataframe['total_bug_desc'].values)
post_seq = tokenizer.texts_to_sequences(dataframe['total_bug_desc'].values)
post_seq_padded = pad_sequences(post_seq, maxlen=MAX_LENGTH)
print('Tokenization done!')

train = post_seq_padded[:-153127]
test = post_seq_padded[-153127:]
print(str(train.shape) + ' ' + str(test.shape))

class_le = LabelEncoder()
dataframe['component_id'] = class_le.fit_transform(dataframe['component_id'])
y = dataframe['component_id'].values
num_class = len(np.unique(y))
y = y[:train.shape[0]]

vocab_size = len(tokenizer.word_index) + 1

EMBEDDING_FILE = 'crawl-300d-2M.vec'
def get_coefs(word, *arr): 
    return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

print('Embeddings made!')

word_index = tokenizer.word_index
nb_words = min(1000000, len(word_index))
embedding_matrix = np.zeros((nb_words, 300))
for word, i in word_index.items():
    if i >= 1000000: 
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector
		
		
def get_model():
    inputs = Input(shape=(MAX_LENGTH, ))
    embedding_layer = Embedding(1000000, 300, input_length=MAX_LENGTH, weights=[embedding_matrix])(inputs)

    #x = LSTM(128)(embedding_layer)
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(embedding_layer)
    x = Dropout(0.3)(GlobalMaxPool1D()(x))
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_class, activation='softmax')(x)
    model = Model(inputs=[inputs], outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    model.summary()
    return model
	
predict = np.zeros((test.shape[0],num_class))

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1256)
ifold = 0

for train_index, test_index in skf.split(train, y):
    kfold_X_train = {}
    kfold_X_valid = {}
    y_train, y_test = y[train_index], y[test_index]
    #for c in ['text','num_vars']:
    kfold_X_train, kfold_X_valid = train[train_index], train[test_index]

    model = get_model()
    
    print('fold: ' + str(ifold))
    #print(model.summary())
    #model = _train_model(model, batch_size, kfold_X_train, y_train, kfold_X_valid, y_test)
    model.fit([kfold_X_train], y=to_categorical(y_train), batch_size=128, epochs=5, verbose=1, validation_data=([kfold_X_valid], to_categorical(y_test)), shuffle=True)
    model.save_weights('model_all_weights_10l_' + str(ifold) + '.h5')
    predictions = model.predict(test, batch_size=128)
    predict += (predictions*0.1)
    #oof_predict[test_index] = model.predict(kfold_X_valid, batch_size=128)
    #cv_score = roc_auc_score(y_test, oof_predict[test_index])
    #scores.append(cv_score)
    #print('score: ',cv_score)
    ifold += 1
    K.clear_session()
	
component_id = np.argmax(predict, axis=1)
component_id = class_le.inverse_transform(component_id)
confidence = np.max(predict, axis=1)

submit = pd.DataFrame({'component_id':component_id, 'confidence':confidence})
submit.to_csv('submit_cudnngru_10folds_10l.csv', index=False)

np.savetxt('10folds_10l', predict, delimiter=',', fmt = '%0.6f')