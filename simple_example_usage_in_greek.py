# INPUT ENA KEIMENO/TEXT - ΠΑΡΑΔΕΙΓΜΑ ΧΡΗΣΗΣ

import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
import nltk
import numpy as np
import spacy
from utils import lemmatize, remove_names, drop_numbers, clean_accent, preprocess_text

#INSERT TEXT#
text = " "
preprocessed_text = preprocess_text(text)

#### Classification ######

# LOAD MODEL #
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import locale
locale.getpreferredencoding = lambda: "UTF-8"
#!pip install transformers
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer,AutoModel,AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
tokenizer = AutoTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
model = AutoModelForSequenceClassification.from_pretrained('nlpaueb/bert-base-greek-uncased-v1', num_labels=2, output_attentions= False, output_hidden_states=False)
model_weights_path= "GREEK_PARAPHRASED.model"
model.load_state_dict(torch.load(model_weights_path))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt",max_length=512).to(device)
outputs = model(**inputs)
logits = outputs.logits
predicted_label = logits.argmax(dim=1).detach().cpu().numpy()

if predicted_label == [1]:
  print("this text contains hate speech")
else:
  print("this text not contains hate speech ")