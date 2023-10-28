
# Import
# Read data

import pandas as pd
import numpy as np

seed = 4
batch_size = 64
epochs = 15
patience = 4
language= "greek" # "greek" or "italian" or "spanish"
mode_train = "no_aug" # no_aug, aug_translated, aug_paraphrased, oversample, aug_all
mode_translated = "none"    # "balanced" or "none"
mode_paraphrased = "none"  # "balanced" or "none"

if (seed == 4):
  seed = 4
  df_train= pd.read_csv("X_train.csv")
  df_val= pd.read_csv("X_val.csv")
  df_test= pd.read_csv("X_test.csv")
if (seed == 17):
  df_train= pd.read_csv("X_train2.csv")
  df_val= pd.read_csv("X_val2.csv")
  df_test= pd.read_csv("X_test2.csv")
if (seed == 31):
  df_train= pd.read_csv("X_train3.csv")
  df_val= pd.read_csv("X_val3.csv")
  df_test= pd.read_csv("X_test3.csv")

# IMPORT TRANSLATED DATA
df_translated = pd.read_csv("translated_greek.csv")

if mode_translated== "balanced" and language == "greek":
  df_translated= df_translated[df_translated["language"]=="italian"]

if mode_translated== "balanced" and language == "italian":
  df_translated= df_translated[df_translated["language"]=="greek"]

if mode_translated== "balanced" and language == "spanish":
  df_translated= df_translated[df_proc["hate"] == 1]
  sample= len(df_train[df_train["hate"]==0]) - len(df_train[df_train["hate"]==1])
  df_translated= df_translated.sample(n=sample, random_state=20)
if mode_translated == "none":
  text_translated = df_translated["text_proc"].astype(str)
  label_translated = df_translated["hate"].astype(int)

# IMPORT PARAPHRASED DATA
hate_for_synth= pd.read_csv("hate_for_synth.csv")
hate_synth_label =hate_for_synth["hate"].astype(int)
nohate_for_synth= pd.read_csv("nohate_for_synth.csv")
nohate_synth_label =nohate_for_synth["hate"].astype(int)
if mode_paraphrased == "none" and (language== "greek" or language=="italian"):
  text_paraphrased=np.concatenate((hate_for_synth["text_proc1"],hate_for_synth["text_proc2"],hate_for_synth["text_proc3"],nohate_for_synth["text_proc1"],nohate_for_synth["text_proc2"],nohate_for_synth["text_proc3"]),axis=0)
  label_paraphrased= np.concatenate((hate_synth_label,hate_synth_label,hate_synth_label,nohate_synth_label,nohate_synth_label,nohate_synth_label),axis=0)
if mode_paraphrased=="none" and language =="spanish":
 text_paraphrased = np.concatenate((hate_for_synth["text_proc1"],hate_for_synth["text_proc2"],hate_for_synth["text_proc3"],hate_for_synth["text_proc4"],hate_for_synth["text_proc5"]),axis=0)
 label_paraphrased = np.concatenate((hate_synth_label,hate_synth_label,hate_synth_label,hate_synth_label,hate_synth_label),axis=0)
if mode_paraphrased == "balanced" and language == "greek":
  text_paraphrased =np.concatenate((hate_for_synth["text_proc1"],hate_for_synth["text_proc2"],hate_for_synth["text_proc3"]),axis=0)
  label_paraphrased = np.concatenate((hate_synth_label,hate_synth_label,hate_synth_label),axis=0)
if mode_paraphrased == "balanced" and language=="italian":
  text_paraphrased=np.concatenate((hate_for_synth["text_proc1"],hate_for_synth["text_proc2"],nohate_for_synth["text_proc1"],nohate_for_synth["text_proc2"],nohate_for_synth["text_proc3"]),axis=0)
  label_paraphrased= np.concatenate((hate_synth_label,hate_synth_label,nohate_synth_label,nohate_synth_label,nohate_synth_label),axis=0)

# LOAD MODEL FAST TEXT
#!pip install fasttext
#!pip install fasttext.util
import fasttext
#import fasttext.util
if language == "greek":
 fasttext.util.download_model('el', if_exists='ignore')
 ft = fasttext.load_model('cc.el.300.bin')
if language == "italian":
 fasttext.util.download_model('it', if_exists='ignore')
 ft = fasttext.load_model('cc.it.300.bin')
if language == "spanish":
 fasttext.util.download_model('es', if_exists='ignore')
 ft = fasttext.load_model('cc.es.300.bin')

#import tensorflow
from tensorflow.random import set_seed
from tensorflow.keras.models import Model
from keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Concatenate,Dense, Activation, Dropout, Flatten,Embedding
from keras.callbacks import EarlyStopping

from keras import metrics

set_seed(seed)

def create_model():
#######
  model= Sequential()
#CREATE INPUT LAYER
  inputs = Input(shape=(128,), dtype='int32', name='inputs')
#CREATE EMBEDDED LAYER
  Embedding1 = Embedding(input_dim = nb_words, output_dim=300 ,weights=[embedding_matrix], input_length=128, trainable=False)(inputs)
# Create the first Conv1D layer
  conv1 = Conv1D(filters=100, kernel_size=2, strides=1, activation='relu')(Embedding1)
  max_pool1 = MaxPooling1D(pool_size=4, strides=4)(conv1)
# Create the second Conv1D layer
  conv2 = Conv1D(filters=100, kernel_size=3, strides=1, activation='relu')(Embedding1)
  max_pool2 = MaxPooling1D(pool_size=4 ,strides=4)(conv2)
# Create the third Conv1D layer
  conv3 = Conv1D(filters=100, kernel_size=4, strides=1, activation='relu')(Embedding1)
  max_pool3 = MaxPooling1D(pool_size=4, strides=4)(conv3)
# Concatenate the outputs of the three Conv1D layers
  concatenated = Concatenate(axis=1)([max_pool1, max_pool2, max_pool3])
# Apply another MaxPooling1D layer
  max_pool_final = MaxPooling1D(pool_size=4, strides=4)(concatenated)
# Flatten the output
  flatten = Flatten()(max_pool_final)
# Add a Dense layer with softmax activation for classification
  dense = Dense(units=2, activation='softmax')(flatten)
# Create the model
  model = Model(inputs=inputs, outputs=dense)
# Compile the model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['Accuracy','Precision','Recall'])
  return model


# load train /dev / test ##
# TRAIN
if mode_train== "no_aug":
  text_train = df_train["text_proc"].astype(str)
  y_train =df_train["hate"].astype(int)
  text_train= text_train.tolist()

if mode_train == "oversample":
  df_hate= df_train[df_train["hate"]== 1]
  df_nohate=df_train[df_train["hate"]== 0]
  text_train = np.concatenate((df_hate["text_proc"],df_hate["text_proc"],df_hate["text_proc"],df_hate["text_proc"],df_hate["text_proc"],df_nohate["text_proc"], axis=0)
  y_train = np.concatenate((df_hate["hate"],df_hate["hate"],df_hate["hate"],df_hate["hate"],df_hate["hate"],df_nohate["hate"],axis=0)
  text_train= text_train.astype(str)
  text_train= text_train.tolist()
  y_train = y_train.astype(int)

if  mode_train== "aug_translated":
  text_train = np.concatenate((df_train["text_proc"], df_translated["text_proc"].astype(str)), axis=0)
  y_train = np.concatenate((df_train["hate"],df_translated["hate"]),axis=0)
  text_train= text_train.astype(str)
  text_train= text_train.tolist()
  y_train = y_train.astype(int)
if mode_train== "aug_paraphrased":
  text_train = np.concatenate((df_train["text_proc"], text_paraphrased), axis=0)
  y_train = np.concatenate((df_train["hate"], label_paraphrased),axis=0)
  text_train= text_train.astype(str)
  text_train= text_train.tolist()
  y_train = y_train.astype(int)

if mode_train == "aug_all":
  text_train = np.concatenate((df_train["text_proc"], df_translated["text_proc"].astype(str),text_paraphrased), axis=0)
  y_train = np.concatenate((df_train["hate"], df_translated["hate"],label_paraphrased),axis=0)
  text_train= text_train.astype(str)
  text_train= text_train.tolist()
  y_train = y_train.astype(int)

# Val
text_val = df_val["text_proc"].astype(str)
text_val= text_val.tolist()
y_val= df_val["hate"].astype(int)

#test
text_test= df_test["text_proc"].astype(str)
text_test= text_test.tolist()
y_test= df_test["hate"].astype(int)

print(len(text_train), len(df_train) ,len(text_train)- sum(y_train),sum(y_train))

#CREATE INDEX OF ALL WORDS THAT ARE IN VOCABULARY
from tensorflow.keras.preprocessing.text import Tokenizer

# Create a tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_train)
word_index = tokenizer.word_index

embed_dim=300

#CREATE EMBEDDING MATRIX FOR EM
print('preparing embedding matrix...')
words_not_found = []
nb_words = len(word_index)
embedding_matrix = np.zeros((nb_words, 300))
for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = ft.get_word_vector(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

max_length=128

#Convert text to integer-encoded sequences
X_train_sequences = tokenizer.texts_to_sequences(text_train)
X_val_sequences =tokenizer.texts_to_sequences(text_val)
X_test_sequences =tokenizer.texts_to_sequences(text_test)
X_train_padded_sequences = pad_sequences(X_train_sequences, maxlen=max_length, padding='post')
X_val_padded_sequences = pad_sequences(X_val_sequences, maxlen=max_length, padding='post')
X_test_padded_sequences = pad_sequences(X_test_sequences, maxlen=max_length, padding='post')

categorical_y_train = to_categorical(y_train)
categorical_y_val= to_categorical(y_val)
categorical_y_test = to_categorical(y_test)

# CREATE MODEL
model = create_model()
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=patience, verbose=1)
callbacks_list = [early_stopping]
h= model.fit(X_train_padded_sequences, categorical_y_train, batch_size=batch_size, epochs=epochs,callbacks=callbacks_list, validation_data=(X_val_padded_sequences, categorical_y_val))

#Evaluate VALIDATION SET
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

y_pred_val = model.predict(X_val_padded_sequences)
y_pred_val = (y_pred_val > 0.5).astype(int)
y_pred_val_single = np.argmax(y_pred_val, axis=1)
y_pred_val_single= np.array(y_pred_val_single)

y_val= categorical_y_val.astype(int)
y_val_single = np.argmax(y_val, axis=1)
y_val_single= np.array(y_val_single)

cm_val = confusion_matrix(y_val_single, y_pred_val_single)
TP = cm_val[1,1]
FP= cm_val[0,1]
TN= cm_val[0,0]
FN= cm_val[1,0]
print(cm_val)

prec_macro_val = precision_score(y_val, y_pred_val,average='macro')
rec_macro_val = recall_score(y_val, y_pred_val,average='macro')
f1_macro_val = f1_score(y_val, y_pred_val,average='macro')
accuracy_val= (TP + TN) / (TP+FP+TN+FN)
precision_val= TP / (TP + FP)
recall_val = TP / (TP + FN)
f1_val= 2*(precision_val*recall_val) / (precision_val+recall_val)
print("VALIDATION SET METRICS /n")
print(accuracy_val,prec_macro_val,rec_macro_val,f1_macro_val,recall_val)
print('Accuracy: {:.2f}'.format(accuracy_val * 100),
                'precisionM: {:.2f}'.format(prec_macro_val*100),
                'recallM: {:.2f}'.format(rec_macro_val*100),
                'f1M: {:.2f}'.format(f1_macro_val*100),
                "precision: {:.2f}".format(precision_val*100),
                "recall: {:.2f}".format(recall_val*100),
                "f1: {:.2f}".format(f1_val*100) )

# EVALUATE TEST SET

y_pred_test = model.predict(X_test_padded_sequences)
y_pred_test = (y_pred_test > 0.5).astype(int)
y_pred_test_single = np.argmax(y_pred_test, axis=1)
y_pred_test_single= np.array(y_pred_test_single)

y_test= categorical_y_test.astype(int)
y_test_single = np.argmax(y_test, axis=1)
y_test_single= np.array(y_test_single)

cm_test = confusion_matrix(y_test_single, y_pred_test_single)
TP = cm_test[1,1]
FP= cm_test[0,1]
TN= cm_test[0,0]
FN= cm_test[1,0]
accuracy_test= (TP + TN) / (TP+FP+TN+FN)
precision_test= TP / (TP + FP)
recall_test = TP / (TP + FN)
f1_test= 2*(precision_test*recall_test) / (precision_test+recall_test)

prec_macro_test = precision_score(y_test, y_pred_test,average='macro')
rec_macro_test = recall_score(y_test, y_pred_test,average='macro')
f1_macro_test = f1_score(y_test, y_pred_test,average='macro')

print(" Test Set Metrics")
print('Accuracy: {:.2f}'.format(accuracy_test * 100),
                'precisionM: {:.2f}'.format(prec_macro_test*100),
                'recallM: {:.2f}'.format(rec_macro_test*100),
                'f1M: {:.2f}'.format(f1_macro_test*100),
                "precision: {:.2f}".format(precision_test*100),
                "recall: {:.2f}".format(recall_test*100),
                "f1: {:.2f}".format(f1_test*100))
