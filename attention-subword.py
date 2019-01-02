##################################
print('running attention subword...')
##################################

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from keras import optimizers, constraints, regularizers, initializers
from keras.engine import InputSpec, Layer
from keras.models import Sequential, Model, load_model
from keras.layers import concatenate, MaxPooling1D, CuDNNGRU, Bidirectional, Dense, Dropout, Activation, Input, Embedding, Flatten, LSTM, SpatialDropout1D, Conv1D, GlobalMaxPool1D, Concatenate
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



###############   attention    #############################

class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Note: The layer has been tested with Keras 2.0.6
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number epsilon to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
    
class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None
			
###################### attention   ###############################################

dataframe = pd.read_csv('train_test_total_bug_desc.csv')
print('dataframe read shape: ' + str(dataframe.shape))

df = dataframe.groupby('component_id')['bug_id'].count().reset_index().sort_values('bug_id', ascending=False)
more_than_one = df[df['bug_id'] > 1]['component_id'].tolist()
#more_than_one = df['component_id'][:1000].tolist()
print(len(more_than_one))
dataframe = dataframe[dataframe['component_id'].isin(more_than_one)]
print('dataframe shape: ' + str(dataframe.shape))

MAX_LENGTH = 300
tokenizer = Tokenizer(num_words=500000)
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

EMBEDDING_FILE = 'crawl-300d-2M-subword.vec'
def get_coefs(word, *arr): 
    return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()


print('Embeddings made!')

word_index = tokenizer.word_index
nb_words = min(500000, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, 300))#np.zeros((nb_words, 300))
for word, i in word_index.items():
    if i >= 500000: 
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector
		
		
def get_model():
    inputs = Input(shape=(MAX_LENGTH, ))
    embedding_layer = Embedding(500000, 300, input_length=MAX_LENGTH, weights=[embedding_matrix])(inputs)

    #x = LSTM(128)(embedding_layer)
    rnn1 = Bidirectional(CuDNNGRU(128, return_sequences=True))(embedding_layer)
    rnn2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(rnn1)
    
    x = concatenate([rnn1, rnn2])
    x = AttentionWeightedAverage()(x)
    #x = GlobalMaxPool1D()(x)
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
    model.fit([kfold_X_train], y=to_categorical(y_train), batch_size=128, epochs=3, verbose=2, validation_data=([kfold_X_valid], to_categorical(y_test)), shuffle=True)
    model.save_weights('model_attention_subword_' + str(ifold) + '.h5')
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
submit.to_csv('submit_attention_subword.csv', index=False)

np.savetxt('attention_subword', predict, delimiter=',', fmt = '%0.6f')