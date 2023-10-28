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
mode_translated ="none"    # "balanced" or "none"
mode_paraphrased = "none"  # "balanced" or "none"
#LOAD DATASETS #
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
if mode_translated == "none":
  text_translated = df_translated["text_proc"].astype(str)
  label_translated = df_translated["hate"].astype(int)
  
if mode_translated== "balanced" and language == "greek":
  df_translated= df_translated[df_translated["language"]=="italian"]

if mode_translated== "balanced" and language == "italian":
  df_translated= df_translated[df_translated["language"]=="greek"]

if mode_translated== "balanced" and language == "spanish":
  df_translated= df_translated[df_proc["hate"] == 1]
  sample= len(df_train[df_train["hate"]==0]) - len(df_train[df_train["hate"]==1])
  df_translated= df_translated.sample(n=sample, random_state=20)


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

# TRAIN #
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

# ENCODE DATA

!pip install --upgrade transformers
import torch
from torch.utils.data import TensorDataset
from transformers import AddedToken, AutoTokenizer, AutoModel,AutoModelWithLMHead,AutoModelForSequenceClassification

if language == "greek":
  tokenizer = AutoTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
if language == "italian":
  tokenizer = AutoTokenizer.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")
if language=="spanish":
  tokenizer = AutoTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')

encoded_data_train = tokenizer.batch_encode_plus( text_train, add_special_tokens=True,
    return_attention_mask=True, pad_to_max_length=True, max_length=128, return_tensors='pt')
encoded_data_val = tokenizer.batch_encode_plus(
    text_val, add_special_tokens=True, return_attention_mask=True, pad_to_max_length=True,
    max_length=128,  return_tensors='pt')
encoded_data_test = tokenizer.batch_encode_plus(
    text_test, add_special_tokens=True, return_attention_mask=True, pad_to_max_length=True,
    max_length=128,  return_tensors='pt')

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(y_train)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(y_val)

input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(y_test)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
## Import model ##
if language == "greek":
  model = AutoModelForSequenceClassification.from_pretrained('nlpaueb/bert-base-greek-uncased-v1',num_labels= 2 , output_attentions= False, output_hidden_states=False)
if language == "italian":
  model = AutoModelForSequenceClassification.from_pretrained('m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0',num_labels= 2 , output_attentions= False, output_hidden_states=False)
if language=="spanish":
  model = AutoModelForSequenceClassification.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased',num_labels= 2 , output_attentions= False, output_hidden_states=False)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

dataloader_train = DataLoader(dataset_train,
                              sampler=RandomSampler(dataset_train),
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val,
                                   sampler=SequentialSampler(dataset_val),
                                   batch_size=batch_size)

from transformers import AdamW, get_linear_schedule_with_warmup
optimizer = AdamW(model.parameters(), lr= lr, eps=1e-8)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(dataloader_train)*epochs)

# METRICS FOR TRAINING EVALUATION
def b_tp(preds, labels):
  #Returns True Positives (TP)
  return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_fp(preds, labels):
  #Returns False Positives (FP)
  return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_tn(preds, labels):
  #Returns True Negatives (TN)
  return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_fn(preds, labels):
  #Returns False Negatives (FN)
  return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_metrics(preds, labels):

  preds = np.argmax(preds, axis = 1).flatten()
  labels = labels.flatten()
  tp = b_tp(preds, labels)
  tn = b_tn(preds, labels)
  fp = b_fp(preds, labels)
  fn = b_fn(preds, labels)
  b_accuracy = (tp + tn) / len(labels)
  b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
  b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
  b_f1 = 2*b_precision* b_recall/(b_precision+b_recall)

  return b_accuracy, b_precision, b_recall, b_f1

#TRAINING PHASE
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def evaluate(dataloader_val):
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []
    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }
        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()
        logits = logits.detach().cpu().numpy()

        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

from tqdm.notebook import tqdm

for epoch in tqdm(range(1, epochs + 1)):

    model.train()

    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:
        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        outputs = model(**inputs)

        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

    torch.save(model.state_dict(), f'finetuned_BERT_epoch_{epoch}.pt')

    tqdm.write(f'\nEpoch {epoch}')

    loss_train_avg = loss_train_total / len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}')

    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    metrics = b_metrics(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'Metrics (Weighted): {metrics}')

_, predictions, true_vals = evaluate(dataloader_validation)
## TEST SET EVALUATION ## 

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve
test_data_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)

# Initialize empty lists to store predictions and true labels
predicted_labels = []
true_labels = []
predicted_probabilities = []
# Set the model to evaluation mode
model.eval()
with torch.no_grad():
    for batch in test_data_loader:
        # Extract batch data
        input_ids_test, attention_masks_test, labels_test = batch
        # Move data to the device (e.g., GPU)
        input_ids_test = input_ids_test.to(device)
        attention_masks_test = attention_masks_test.to(device)
        labels_test = labels_test.to(device)

        # Forward pass through the model
        outputs = model(input_ids=input_ids_test, attention_mask=attention_masks_test)
        logits = outputs.logits
        batch_predicted_labels = logits.argmax(dim=1).detach().cpu().numpy()
        batch_true_labels = labels_test.detach().cpu().numpy()
        batch_predicted_probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        # Append batch results to the overall lists
        predicted_labels.extend(batch_predicted_labels)
        true_labels.extend(batch_true_labels)
        predicted_probabilities.extend(batch_predicted_probabilities)
# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
roc_auc = roc_auc_score(true_labels, predicted_probabilities)



fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities)
precision_macro = precision_score(true_labels, predicted_labels,average='macro')
recall_macro = recall_score(true_labels, predicted_labels,average='macro')
f1_macro = f1_score(true_labels, predicted_labels,average='macro')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')
print(f'Precision(M): {precision_macro:.4f}')
print(f'Recall(M): {recall_macro:.4f}')
print(f'F1 Score(M): {f1_macro:.4f}')
