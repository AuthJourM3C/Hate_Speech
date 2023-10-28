# METATROPI VIDEO SE AUDIO
#!pip install pytube
from pytube import YouTube
import moviepy.editor as mp
video_url = 'https://www.youtube.com/'
yt = YouTube(video_url)
stream = yt.streams.filter(only_audio=True).first()
# Download the audio stream as an MP4 file
audio_filename = 'audio.mp4'
stream.download(output_path='.', filename=audio_filename)

# Load the audio file using moviepy
audio = mp.AudioFileClip(audio_filename)

# ΓΙΑ ΜΕΓΑΛΑ ΒΙΝΤΕΟ ΧΡΕΙΑΣΤΗΚΕ ΝΑ ΦΟΡΤΩΘΟΥΝ ΣΤΟ GOOGLE CLOUD και να γίνει από εκεί η μετατροπή του ήχου σε κείμενο #

#SPEECH TO TEXT #

#!pip install --upgrade google-cloud-speech
from google.cloud import speech
from google.oauth2 import service_account
client = speech.SpeechClient.from_service_account_file("triple-shadow.json")
gcs_uri = "gs://####.mp3"
audio_file= speech.RecognitionAudio(uri=gcs_uri)
config= speech.RecognitionConfig(sample_rate_hertz=44100 ,enable_automatic_punctuation=True,language_code="el-GR",enable_word_time_offsets=True)
print("Waiting for operation to complete...")
operation=client.long_running_recognize(config=config, audio=audio_file)
# Iterate through the results to get the transcribed text
response = operation.result()
# for a single text #
# Recognize
#for result in response.results:
#    for word_info in result.alternatives[0].words:
#        print(
#            f"Word: {word_info.word}, Start Time: {word_info.start_time.total_seconds()}, End Time: {word_info.end_time.total_seconds()}"
#        )
import pandas as pd
# SAVE WHEN EACH WORD APPEARS
df_words = pd.DataFrame(columns=["word", "time_start", "time_end"])
for result in response.results:
    alternative = result.alternatives[0]
    print("Transcript: {}".format(alternative.transcript))
    # Print word time offsets
    for word_info in alternative.words:
          new_row = {"word":word_info.word, "time_start": word_info.start_time.total_seconds(), "time_end": word_info.end_time.total_seconds()}
          df_words = df_words.append(new_row, ignore_index=True)

# segmentation #

import math
text= " "  # THE WHOLE TRANSCRIPT
size_segment= 30 # test hate_speech for each 30 words
word_length= len(df_words) # length
segments = math.ceil(word_length/ size_segment)
data1 = {'text': ([" "]*segments), 'time_start': 0*segments, "time_end": 0*segments, "hate": 0*segments}
df = pd.DataFrame(data1)

for i in range(word_length):
  current_segment= math.ceil((i+1)/size_segment)
  df.loc[current_segment-1,'text'] += ' ' + df_words.loc[i,"word"]
  if ((i+1)%size_segment == 0) and i!=word_length:
    df.loc[current_segment-1,"time_end"]= df_words.loc[i,"time_end"]
    df.loc[current_segment,"time_start"]= df_words.loc[i,"time_end"]
  text += ' ' + df_words.loc[i,"word"]
df.loc[current_segment-1,"time_end"]= df_words.loc[i,"time_end"]

# Certain essential functions for preprocessing...
regexp = RegexpTokenizer('\w+')
!python -m spacy download el_core_news_lg

# preprocessing text
from utils import lemmatize, remove_names, drop_numbers, clean_accent, preprocess_text
df["text_proc"]= df["text"].apply(lambda x: preprocess_text(x))

#### Classification ######

# LOAD MODEL #
import locale
locale.getpreferredencoding = lambda: "UTF-8"
#!pip install transformers
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer,AutoModel,AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

tokenizer = AutoTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
model = AutoModelForSequenceClassification.from_pretrained('nlpaueb/bert-base-greek-uncased-v1', num_labels=2, output_attentions= False, output_hidden_states=False)
model_weights_path= "/content/drive/MyDrive/ΔΙΠΛΩΜΑΤΙΚΗ/Datasets/Fine_Tune_Bert/GREEK_PARAPHRASED.model"
model.load_state_dict(torch.load(model_weights_path))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Encode Data #
input_text = df["text_proc"].tolist()
y_test= df["hate"].tolist()
model.eval()
inputs = tokenizer(input_text, add_special_tokens=True, return_attention_mask=True, pad_to_max_length=True,  return_tensors='pt').to(device)
input_ids = inputs['input_ids'].to(device)
attention_masks = inputs['attention_mask'].to(device)
labels_test = torch.tensor(y_test).to(device)

# Predict #
outputs = model(**inputs.to(device))
logits = outputs.logits
predicted_labels = logits.argmax(dim=1).detach().cpu().numpy()

df["hate"] = predicted_labels
for i in df.index:
   if df.loc[i,"hate"] == 1 :
     print("Το κείμενο το οποίο βρίσκεται στη διάρκεια [",df.loc[i,"time_start"],":",df.loc[i,"time_end"],"] περιέχει ρητορική μίσους")
     print("Ακουλουθεί το επίμαχο κείμενο:")
     print(df.loc[i,"text"])

