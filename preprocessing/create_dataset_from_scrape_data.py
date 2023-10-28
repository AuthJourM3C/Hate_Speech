import numpy as np
import pandas as pd

English_Hate_Negative= pd.read_csv('English_Hate_Negative.csv', header= None )
English_Hate_Neutral= pd.read_csv('English_Hate_Neutral.csv', header=None)
English_Hate_Positive= pd.read_csv('English_Hate_Positive.csv', header=None)
English_NoHate_Negative= pd.read_csv('English_NoHate_Negative.csv', header=None)
English_NoHate_Neutral= pd.read_csv('/English_NoHate_Neutral.csv', header=None)
English_NoHate_Positive= pd.read_csv('/English_NoHate_Positive.csv', header=None)
Greek_Hate_Negative= pd.read_csv('Greek_Hate_Negative.csv', header=None)
Greek_Hate_Neutral= pd.read_csv('Greek_Hate_Neutral.csv', header=None)
Greek_Hate_Positive= pd.read_csv('Greek_Hate_Positive.csv', header=None)
Greek_NoHate_Negative= pd.read_csv('Greek_NoHate_Negative.csv', header=None)
Greek_NoHate_Neutral= pd.read_csv('Greek_NoHate_Neutral.csv', header=None)
Greek_NoHate_Positive= pd.read_csv('Greek_NoHate_Positive.csv', header=None)
Italian_Hate_Negative= pd.read_csv('Italian_Hate_Negative.csv', header=None)
Italian_Hate_Neutral= pd.read_csv('Italian_Hate_Neutral.csv', header=None)
Italian_Hate_Positive= pd.read_csv("Italian_Hate_Positive.csv", header=None)
Italian_NoHate_Negative= pd.read_csv('Italian_NoHate_Negative.csv', header=None)
Italian_NoHate_Neutral= pd.read_csv('Italian_NoHate_Neutral.csv', header=None)
Italian_NoHate_Positive= pd.read_csv('Italian_NoHate_Positive.csv', header=None)
Other_Hate_Negative= pd.read_csv('Other_Hate_Negative.csv', header=None)
Other_Hate_Neutral=pd.read_csv('Other_Hate_Neutral.csv', header= None )
Other_Hate_Positive=pd.read_csv('Other_Hate_Positive.csv', header= None )
Other_NoHate_Negative=pd.read_csv('Other_NoHate_Negative.csv', header= None )
Other_NoHate_Neutral=pd.read_csv('Other_NoHate_Neutral.csv', header= None )
Other_NoHate_Positive=pd.read_csv('Other_NoHate_Positive.csv', header= None )
Spanish_Hate_Negative=pd.read_csv('Spanish_Hate_Negative.csv', header= None )
Spanish_Hate_Neutral=pd.read_csv('Spanish_Hate_Neutral.csv', header= None )
Spanish_Hate_Positive=pd.read_csv('Spanish_Hate_Positive.csv', header= None )
Spanish_NoHate_Negative=pd.read_csv('Spanish_NoHate_Negative.csv', header= None )
Spanish_NoHate_Neutral=pd.read_csv('Spanish_NoHate_Neutral.csv', header= None )
Spanish_NoHate_Positive=pd.read_csv('Spanish_NoHate_Positive.csv', header= None )

#English_Hate_Negative
length1= len(English_Hate_Negative.index)
hate1=[1]*length1
sentiment1=['negative']* length1
language1= ["english"]*length1
English_Hate_Negative= English_Hate_Negative.rename(columns={0:"text"})
English_Hate_Negative["language"]= language1
English_Hate_Negative["sentiment"]= sentiment1
English_Hate_Negative["hate"]= hate1
print(English_Hate_Negative.head())
#English_Hate_Neutral
length1= len(English_Hate_Neutral.index)
hate1=[1]*length1
sentiment1=['neutral']* length1
language1= ["english"]*length1
English_Hate_Neutral= English_Hate_Neutral.rename(columns={0:"text"})
English_Hate_Neutral["language"]= language1
English_Hate_Neutral["sentiment"]= sentiment1
English_Hate_Neutral["hate"]= hate1
print(English_Hate_Neutral.head())
#English_Hate_Positive
length1= len(English_Hate_Positive.index)
hate1=[1]*length1
sentiment1=['positive']* length1
language1= ["english"]*length1
English_Hate_Positive= English_Hate_Positive.rename(columns={0:"text"})
English_Hate_Positive["language"]= language1
English_Hate_Positive["sentiment"]= sentiment1
English_Hate_Positive["hate"]= hate1
print(English_Hate_Positive.head())
#English_NoHate_Negative
length1= len(English_NoHate_Negative.index)
hate1=[0]*length1
sentiment1=['negative']* length1
language1= ["english"]*length1
English_NoHate_Negative= English_NoHate_Negative.rename(columns={0:"text"})
English_NoHate_Negative["language"]= language1
English_NoHate_Negative["sentiment"]= sentiment1
English_NoHate_Negative["hate"]= hate1
print(English_NoHate_Negative.head())
#English_NoHate_Neutral
length1= len(English_NoHate_Neutral.index)
hate1=[0]*length1
sentiment1=['neutral']* length1
language1= ["english"]*length1
English_NoHate_Neutral= English_NoHate_Neutral.rename(columns={0:"text"})
English_NoHate_Neutral["language"]= language1
English_NoHate_Neutral["sentiment"]= sentiment1
English_NoHate_Neutral["hate"]= hate1
print(English_NoHate_Neutral.head())
#English_NoHate_Positive
length1= len(English_NoHate_Positive.index)
hate1=[0]*length1
sentiment1=['positive']* length1
language1= ["english"]*length1
English_NoHate_Positive= English_NoHate_Positive.rename(columns={0:"text"})
English_NoHate_Positive["language"]= language1
English_NoHate_Positive["sentiment"]= sentiment1
English_NoHate_Positive["hate"]= hate1
print(English_NoHate_Positive.head())

frames = [English_Hate_Negative,English_Hate_Neutral,English_Hate_Positive,English_NoHate_Negative,English_NoHate_Neutral,English_NoHate_Positive]
English = pd.concat(frames)
English.to_csv("C:/Users/thana/Desktop/Pharm Dataset/final/English.csv",header=True,index=False)

#Greek_Hate_Negative
length1= len(Greek_Hate_Negative.index)
hate1=[1]*length1
sentiment1=['negative']* length1
language1= ["greek"]*length1
Greek_Hate_Negative= Greek_Hate_Negative.rename(columns={0:"text"})
Greek_Hate_Negative["language"]= language1
Greek_Hate_Negative["sentiment"]= sentiment1
Greek_Hate_Negative["hate"]= hate1
print(Greek_Hate_Negative.head())
#Greek_Hate_Neutral
length1= len(Greek_Hate_Neutral.index)
hate1=[1]*length1
sentiment1=['neutral']* length1
language1= ["greek"]*length1
Greek_Hate_Neutral= Greek_Hate_Neutral.rename(columns={0:"text"})
Greek_Hate_Neutral["language"]= language1
Greek_Hate_Neutral["sentiment"]= sentiment1
Greek_Hate_Neutral["hate"]= hate1
print(Greek_Hate_Neutral.head())
#Greek_Hate_Positive
length1= len(Greek_Hate_Positive.index)
hate1=[1]*length1
sentiment1=['positive']* length1
language1= ["greek"]*length1
Greek_Hate_Positive= Greek_Hate_Positive.rename(columns={0:"text"})
Greek_Hate_Positive["language"]= language1
Greek_Hate_Positive["sentiment"]= sentiment1
Greek_Hate_Positive["hate"]= hate1
print(Greek_Hate_Positive.head())
#Greek_NoHate_Negative
length1= len(Greek_NoHate_Negative.index)
hate1=[0]*length1
sentiment1=['negative']* length1
language1= ["greek"]*length1
Greek_NoHate_Negative= Greek_NoHate_Negative.rename(columns={0:"text"})
Greek_NoHate_Negative["language"]= language1
Greek_NoHate_Negative["sentiment"]= sentiment1
Greek_NoHate_Negative["hate"]= hate1
print(Greek_NoHate_Negative.head())
#Greek_NoHate_Neutral
length1= len(Greek_NoHate_Neutral.index)
hate1=[0]*length1
sentiment1=['neutral']* length1
language1= ["greek"]*length1
Greek_NoHate_Neutral= Greek_NoHate_Neutral.rename(columns={0:"text"})
Greek_NoHate_Neutral["language"]= language1
Greek_NoHate_Neutral["sentiment"]= sentiment1
Greek_NoHate_Neutral["hate"]= hate1
print(Greek_NoHate_Neutral.head())
#Greek_NoHate_Positive
length1= len(Greek_NoHate_Positive.index)
hate1=[0]*length1
sentiment1=['positive']* length1
language1= ["greek"]*length1
Greek_NoHate_Positive= Greek_NoHate_Positive.rename(columns={0:"text"})
Greek_NoHate_Positive["language"]= language1
Greek_NoHate_Positive["sentiment"]= sentiment1
Greek_NoHate_Positive["hate"]= hate1
print(Greek_NoHate_Positive.head())
frames= [Greek_Hate_Negative,Greek_Hate_Neutral,Greek_Hate_Positive,Greek_NoHate_Negative,Greek_NoHate_Neutral,Greek_NoHate_Positive]
greek = pd.concat(frames)
greek.to_csv("C:/Users/thana/Desktop/Pharm Dataset/final/Greek.csv",header=True,index=False)

#Other_Hate_Negative
length1= len(Other_Hate_Negative.index)
hate1=[1]*length1
sentiment1=['negative']* length1
language1= ["other"]*length1
Other_Hate_Negative= Other_Hate_Negative.rename(columns={0:"text"})
Other_Hate_Negative["language"]= language1
Other_Hate_Negative["sentiment"]= sentiment1
Other_Hate_Negative["hate"]= hate1
print(Other_Hate_Negative.head())
#Other_Hate_Neutral
length1= len(Other_Hate_Neutral.index)
hate1=[1]*length1
sentiment1=['neutral']* length1
language1= ["other"]*length1
Other_Hate_Neutral= Other_Hate_Neutral.rename(columns={0:"text"})
Other_Hate_Neutral["language"]= language1
Other_Hate_Neutral["sentiment"]= sentiment1
Other_Hate_Neutral["hate"]= hate1
print(Other_Hate_Neutral.head())
#Other_Hate_Positive
length1= len(Other_Hate_Positive.index)
hate1=[1]*length1
sentiment1=['positive']* length1
language1= ["other"]*length1
Other_Hate_Positive= Other_Hate_Positive.rename(columns={0:"text"})
Other_Hate_Positive["language"]= language1
Other_Hate_Positive["sentiment"]= sentiment1
Other_Hate_Positive["hate"]= hate1
print(Other_Hate_Positive.head())
#Other_NoHate_Negative
length1= len(Other_NoHate_Negative.index)
hate1=[0]*length1
sentiment1=['negative']* length1
language1= ["other"]*length1
Other_NoHate_Negative= Other_NoHate_Negative.rename(columns={0:"text"})
Other_NoHate_Negative["language"]= language1
Other_NoHate_Negative["sentiment"]= sentiment1
Other_NoHate_Negative["hate"]= hate1
print(Other_NoHate_Negative.head())
#Other_NoHate_Neutral
length1= len(Other_NoHate_Neutral.index)
hate1=[0]*length1
sentiment1=['neutral']* length1
language1= ["other"]*length1
Other_NoHate_Neutral= Other_NoHate_Neutral.rename(columns={0:"text"})
Other_NoHate_Neutral["language"]= language1
Other_NoHate_Neutral["sentiment"]= sentiment1
Other_NoHate_Neutral["hate"]= hate1
print(Other_NoHate_Neutral.head())
#Other_NoHate_Positive
length1= len(Other_NoHate_Positive.index)
hate1=[0]*length1
sentiment1=['positive']* length1
language1= ["other"]*length1
Other_NoHate_Positive= Other_NoHate_Positive.rename(columns={0:"text"})
Other_NoHate_Positive["language"]= language1
Other_NoHate_Positive["sentiment"]= sentiment1
Other_NoHate_Positive["hate"]= hate1
print(Other_NoHate_Positive.head())
frames= [Other_Hate_Negative,Other_Hate_Neutral,Other_Hate_Positive,Other_NoHate_Negative,Other_NoHate_Neutral,Other_NoHate_Positive]
other = pd.concat(frames)
other.to_csv("C:/Users/thana/Desktop/Pharm Dataset/final/other.csv",header=True,index=False)

#Italian_Hate_Negative
length1= len(Italian_Hate_Negative.index)
hate1=[1]*length1
sentiment1=['negative']* length1
language1= ["italian"]*length1
Italian_Hate_Negative= Italian_Hate_Negative.rename(columns={0:"text"})
Italian_Hate_Negative["language"]= language1
Italian_Hate_Negative["sentiment"]= sentiment1
Italian_Hate_Negative["hate"]= hate1
print(Italian_Hate_Negative.head())
#Italian_Hate_Neutral
length1= len(Italian_Hate_Neutral.index)
hate1=[1]*length1
sentiment1=['neutral']* length1
language1= ["italian"]*length1
Italian_Hate_Neutral= Italian_Hate_Neutral.rename(columns={0:"text"})
Italian_Hate_Neutral["language"]= language1
Italian_Hate_Neutral["sentiment"]= sentiment1
Italian_Hate_Neutral["hate"]= hate1
print(Italian_Hate_Neutral.head())
#Italian_Hate_Positive
length1= len(Italian_Hate_Positive.index)
hate1=[1]*length1
sentiment1=['positive']* length1
language1= ["italian"]*length1
Italian_Hate_Positive= Italian_Hate_Positive.rename(columns={0:"text"})
Italian_Hate_Positive["language"]= language1
Italian_Hate_Positive["sentiment"]= sentiment1
Italian_Hate_Positive["hate"]= hate1
print(Italian_Hate_Positive.head())
#Italian_NoHate_Negative
length1= len(Italian_NoHate_Negative.index)
hate1=[0]*length1
sentiment1=['negative']* length1
language1= ["italian"]*length1
Italian_NoHate_Negative= Italian_NoHate_Negative.rename(columns={0:"text"})
Italian_NoHate_Negative["language"]= language1
Italian_NoHate_Negative["sentiment"]= sentiment1
Italian_NoHate_Negative["hate"]= hate1
print(Italian_NoHate_Negative.head())
#Italian_NoHate_Neutral
length1= len(Italian_NoHate_Neutral.index)
hate1=[0]*length1
sentiment1=['neutral']* length1
language1= ["italian"]*length1
Italian_NoHate_Neutral= Italian_NoHate_Neutral.rename(columns={0:"text"})
Italian_NoHate_Neutral["language"]= language1
Italian_NoHate_Neutral["sentiment"]= sentiment1
Italian_NoHate_Neutral["hate"]= hate1
print(Italian_NoHate_Neutral.head())
#Italian_NoHate_Positive
length1= len(Italian_NoHate_Positive.index)
hate1=[0]*length1
sentiment1=['positive']* length1
language1= ["italian"]*length1
Italian_NoHate_Positive= Italian_NoHate_Positive.rename(columns={0:"text"})
Italian_NoHate_Positive["language"]= language1
Italian_NoHate_Positive["sentiment"]= sentiment1
Italian_NoHate_Positive["hate"]= hate1
print(Italian_NoHate_Positive.head())
frames= [Italian_Hate_Negative,Italian_Hate_Neutral,Italian_Hate_Positive,Italian_NoHate_Negative,Italian_NoHate_Neutral,Italian_NoHate_Positive]
italian = pd.concat(frames)
italian.to_csv("C:/Users/thana/Desktop/Pharm Dataset/final/italian.csv",header=True,index=False)

#Spanish_Hate_Negative
length1= len(Spanish_Hate_Negative.index)
hate1=[1]*length1
sentiment1=['negative']* length1
language1= ["spanish"]*length1
Spanish_Hate_Negative= Spanish_Hate_Negative.rename(columns={0:"text"})
Spanish_Hate_Negative["language"]= language1
Spanish_Hate_Negative["sentiment"]= sentiment1
Spanish_Hate_Negative["hate"]= hate1
print(Spanish_Hate_Negative.head())
#Spanish_Hate_Neutral
length1= len(Spanish_Hate_Neutral.index)
hate1=[1]*length1
sentiment1=['neutral']* length1
language1= ["spanish"]*length1
Spanish_Hate_Neutral= Spanish_Hate_Neutral.rename(columns={0:"text"})
Spanish_Hate_Neutral["language"]= language1
Spanish_Hate_Neutral["sentiment"]= sentiment1
Spanish_Hate_Neutral["hate"]= hate1
print(Spanish_Hate_Neutral.head())
#Spanish_Hate_Positive
length1= len(Spanish_Hate_Positive.index)
hate1=[1]*length1
sentiment1=['positive']* length1
language1= ["spanish"]*length1
Spanish_Hate_Positive= Spanish_Hate_Positive.rename(columns={0:"text"})
Spanish_Hate_Positive["language"]= language1
Spanish_Hate_Positive["sentiment"]= sentiment1
Spanish_Hate_Positive["hate"]= hate1
print(Spanish_Hate_Positive.head())
#Spanish_NoHate_Negative
length1= len(Spanish_NoHate_Negative.index)
hate1=[0]*length1
sentiment1=['negative']* length1
language1= ["spanish"]*length1
Spanish_NoHate_Negative= Spanish_NoHate_Negative.rename(columns={0:"text"})
Spanish_NoHate_Negative["language"]= language1
Spanish_NoHate_Negative["sentiment"]= sentiment1
Spanish_NoHate_Negative["hate"]= hate1
print(Spanish_NoHate_Negative.head())
#Spanish_NoHate_Neutral
length1= len(Spanish_NoHate_Neutral.index)
hate1=[0]*length1
sentiment1=['neutral']* length1
language1= ["spanish"]*length1
Spanish_NoHate_Neutral= Spanish_NoHate_Neutral.rename(columns={0:"text"})
Spanish_NoHate_Neutral["language"]= language1
Spanish_NoHate_Neutral["sentiment"]= sentiment1
Spanish_NoHate_Neutral["hate"]= hate1
print(Spanish_NoHate_Neutral.head())
#Spanish_NoHate_Positive
length1= len(Spanish_NoHate_Positive.index)
hate1=[0]*length1
sentiment1=['positive']* length1
language1= ["spanish"]*length1
Spanish_NoHate_Positive= Spanish_NoHate_Positive.rename(columns={0:"text"})
Spanish_NoHate_Positive["language"]= language1
Spanish_NoHate_Positive["sentiment"]= sentiment1
Spanish_NoHate_Positive["hate"]= hate1
print(Spanish_NoHate_Positive.head())
frames= [Spanish_Hate_Negative,Spanish_Hate_Neutral,Spanish_Hate_Positive,Spanish_NoHate_Negative,Spanish_NoHate_Neutral,Spanish_NoHate_Positive]
spanish = pd.concat(frames)
spanish.to_csv("C:/Users/thana/Desktop/Pharm Dataset/final/spanish.csv",header=True,index=False)

frames= [English,greek, italian, other,spanish]
Final_Dataset= pd.concat(frames)
Final_Dataset.to_csv("C:/Users/thana/Desktop/Pharm Dataset/final/Final_Dataset.csv",header=True,index=False)