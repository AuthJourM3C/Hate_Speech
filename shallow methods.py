# Import
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import recall_score,f1_score,accuracy_score,precision_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.svm import SVC
# Import
# Read data
seed = 4 # 4 or 17 or 31
batch_size = 64
epochs = 15
patience = 4
language= "greek" # "greek" or "italian" or "spanish"
mode_train = "no_aug" # no_aug, aug_translated, aug_paraphrased, oversample, aug_all
mode_translated ="none"    # "balanced" or "none"
mode_paraphrased = "none"  # "balanced" or "none"
train_model = "SVM" , # SVM or LR or RF or NB

if (seed == 4):
  df_train= pd.read_csv("X_train.csv")
  df_val= pd.read_csv("X_val    .csv")
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

# load train /dev / test ##
# TRAIN
if mode_train== "no_aug":
  X_train = df_train["text_proc"].astype(str)
  y_train =df_train["hate"].astype(int)
  text_train= X_train.tolist()
if mode_train == "oversample":
  df_hate= df_train[df_train["hate"]== 1]
  df_nohate=df_train[df_train["hate"]== 0]
  X_train_final = np.concatenate((df_hate["text_proc"],df_hate["text_proc"],df_hate["text_proc"],df_hate["text_proc"],df_hate["text_proc"],df_nohate["text_proc"]), axis=0)
  y_train_final = np.concatenate((df_hate["hate"],df_hate["hate"],df_hate["hate"],df_hate["hate"],df_hate["hate"],df_nohate["hate"]),axis=0)
  X_train_final= X_train_final.astype(str)
  text_train= X_train_final.tolist()
  y_train = y_train_final.astype(int)
if  mode_train== "aug_translated":
  X_train_final = np.concatenate((df_train["text_proc"], text_translated), axis=0)
  y_train_final = np.concatenate((df_train["hate"], label_translated),axis=0)
  X_train_final= X_train_final.astype(str)
  text_train= X_train_final.tolist()
  y_train = y_train_final.astype(int)
if mode_train== "aug_paraphrased":
  X_train_final = np.concatenate((df_train["text_proc"],text_paraphrased), axis=0)
  y_train_final = np.concatenate((df_train["hate"],label_paraphrased),axis=0)
  X_train_final= X_train_final.astype(str)
  text_train= X_train_final.tolist()
  y_train = y_train_final.astype(int)
if mode_train == "aug_all":
  X_train_final = np.concatenate((df_train["text_proc"], text_translated,text_paraphrased), axis=0)
  y_train_final = np.concatenate((df_train["hate"], label_translated,label_paraphrased),axis=0)
  X_train_final= X_train_final.astype(str)
  text_train= X_train_final.tolist()
  y_train = y_train_final.astype(int)
# Val
X_val = df_val["text_proc"].astype(str)
text_val= X_val.tolist()
y_val= df_val["hate"].astype(int)
#test
X_test= df_test["text_proc"].astype(str)
text_test= X_test.tolist()
y_test=df_test["hate"].astype(int)


##### CLASSIFIERS ####
#NAIVE BAYES
if train_model == "NB":
    alpha = 1
    vectorizer =CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
    X_train_vectors = vectorizer.fit_transform(text_train)
    X_test_vectors = vectorizer.transform(X_test)
    model = BernoulliNB(alpha= alpha)
    model.fit(X_train_vectors, y_train)
#RANDOM FOREST
if train_model== "RF":
    depth=140
    n_estimators=700
    vectorizer =TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
    X_train_vectors = vectorizer.fit_transform(text_train)
    X_test_vectors = vectorizer.transform(X_test)
    model = RandomForestClassifier(n_estimators= n_estimators, min_samples_split=5, min_samples_leaf=1, max_depth=depth, random_state=42)
    model.fit(X_train_vectors, y_train)
#SVM
if train_model == "SVM":
    # SVM
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
    X_train_vectors = vectorizer.fit_transform(text_train)
    X_test_vectors = vectorizer.transform(X_test)
    clf = SVC(kernel='linear', C= 1)
    clf.fit(X_train_vectors, y_train)

# LOGISTIC REGRESSION
if train_model == "LR":
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000) # or CountVectorizer for BOW
X_train_vectors = vectorizer.fit_transform(text_train)
X_test_vectors = vectorizer.transform(X_test)
model = LogisticRegression()
model.fit(X_train_vectors, y_train)

# CALCULATE METRICS 
y_pred = model.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)
precision= precision_score(y_test,y_pred)
recall= recall_score(y_test, y_pred)
f1= f1_score(y_test, y_pred)
print("Accuracy: {:.2f}".format(accuracy * 100),"precision: {:.2f}".format(precision * 100),"recall: {:.2f}".format(recall * 100),"f1_score: {:.2f}".format(f1 * 100))
