import re
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import spacy
!python -m spacy download el_core_news_lg #or es_core_news_lg or it_core_news_lg
spacy.require_gpu()
nlp=spacy.load("el_core_news_lg")
regexp = RegexpTokenizer('\w+')

def lemmatize(text):
    x= nlp(str(text))
    x_lemma = ' '.join([text.lemma_ for text in x])
    return x_lemma

def remove_names(x):
    for word in x.split():
        if word[0] == "@":
            x = x.replace(word, "")
    return x
def drop_numbers(text):
    text_without_numbers = re.sub(r'\d+', '', text)
    return text_without_numbers
def clean_accent(text):

    t = str(text)

    # el
    t = t.replace('Ά', 'Α')
    t = t.replace('Έ', 'Ε')
    t = t.replace('Ί', 'Ι')
    t = t.replace('Ή', 'Η')
    t = t.replace('Ύ', 'Υ')
    t = t.replace('Ό', 'Ο')
    t = t.replace('Ώ', 'Ω')
    t = t.replace('ά', 'α')
    t = t.replace('έ', 'ε')
    t = t.replace('ί', 'ι')
    t = t.replace('ή', 'η')
    t = t.replace('ύ', 'υ')
    t = t.replace('ό', 'ο')
    t = t.replace('ώ', 'ω')
    t = t.replace('ς', 'σ')

    # es
    t = t.replace('Á', 'A')
    t = t.replace('É', 'E')
    t = t.replace('Í', 'I')
    t = t.replace('Ñ', 'N')
    t = t.replace('Ó', 'O')
    t = t.replace('Ú', 'U')
    t = t.replace('Ü', 'U')
    t = t.replace('á', 'a')
    t = t.replace('é', 'e')
    t = t.replace('í', 'i')
    t = t.replace('ñ', 'n')
    t = t.replace('ó', 'o')
    t = t.replace('ú', 'u')
    t = t.replace('ü', 'u')

    # it
    t = t.replace('À', 'A')
    t = t.replace('È', 'E')
    t = t.replace('É', 'E')
    t = t.replace('Ì', 'I')
    t = t.replace('Ò', 'O')
    t = t.replace('Ó', 'O')
    t = t.replace('Ù', 'U')
    t = t.replace('à', 'a')
    t = t.replace('è', 'e')
    t = t.replace('é', 'e')
    t = t.replace('ì', 'i')
    t = t.replace('ò', 'o')
    t = t.replace('ó', 'o')
    t = t.replace('ù', 'u')

    return t

stop= pd.read_csv("/content/drive/MyDrive/ΔΙΠΛΩΜΑΤΙΚΗ/Datasets/stopwords/stopwords__greek.csv", header=None )
stop= stop.values.tolist()
stopw = [item for sublist in stop for item in sublist]
stopwords=set(stopw)

#Lemmatization
df["text_lemma"]= np.empty(len(df.index),dtype=object)
for i in df.index:
    x = str(df.loc[i, "text"])
    df.loc[i,"text_lemma"]= lemmatize(x)
#clean accent
df["text_proc"]=df["text"].apply(lambda x: clean_accent(x))
#lower
df['text_proc'] = df['text_proc'].astype(str).str.lower()
#Remove Links
df["text_proc"]= df["text_proc"].apply(lambda x: re.sub(r'https?:\/\/.*[\r\n]*', '', x))
#Remove Hastags
df["text_proc"]= df["text_proc"].apply(lambda x: re.sub(r'#', '', x))
#Remove names
df["text_proc"]=df["text_proc"].apply(lambda x: re.sub(r'@[A-Za-z0-9_]+', '', x))
#Remove Punctuation
df['text_proc'] = df['text_proc'].apply(lambda x: re.sub(r'[^\w\s]','',x))
#Remove number
df['text_proc'] = df['text_proc'].apply(lambda x:drop_numbers(x))
#Tokenize
df['text_token']=df['text_proc'].apply(regexp.tokenize)
#Removes Stop Words
df['text_token'] = df['text_token'].apply(lambda x: [item for item in x if item not in stopwords])
df['text_proc'] = df['text_token'].apply(lambda x: ' '.join([item for item in x if len(item)>3])) #Κραταω λέξεις με πάνω απο 2 γράμματα
df["text_proc"]= df["text_proc"].astype(str)
df["text_proc"]= df["text_proc"].tolist()
#Remove empty rows
df.dropna(how='any', inplace=True)
df = df[df['text_token'].astype(bool)]
df = df.drop(columns=["text_lemma","text_token"])
