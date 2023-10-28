
df = pd.read_csv("greek.csv")  # or greek or italian

# Get 500 samples of hate and 500 samples of no hate
df_hate = df[df["hate"] == 1]
df_nohate = df[df["hate"] == 0]

df_hate_len = df_train_hate[df_train_hate["text"].str.len() < 1000]
df_nohate_len = df_train_nohate[df_train_nohate["text"].str.len() < 1000]

hate_for_synth = df_train_hate_len.sample(n=500, random_state=20)
nohate_for_synth = df_train_nohate_len.sample(n=500, random_state=25)

# prepare the dataframe for saving the paraphrased texts
hate_for_synth["text1"] = ['empty'] * hate_for_synth.shape[0]
hate_for_synth["text2"] = ['empty'] * hate_for_synth.shape[0]
hate_for_synth["text3"] = ['empty'] * hate_for_synth.shape[0]
hate_for_synth["text4"] = ['empty'] * hate_for_synth.shape[0]
hate_for_synth["text5"] = ['empty'] * hate_for_synth.shape[0]

nohate_for_synth["text1"] = ['empty'] * nohate_for_synth.shape[0]
nohate_for_synth["text2"] = ['empty'] * nohate_for_synth.shape[0]
nohate_for_synth["text3"] = ['empty'] * nohate_for_synth.shape[0]
nohate_for_synth["text4"] = ['empty'] * hate_for_synth.shape[0]
nohate_for_synth["text5"] = ['empty'] * hate_for_synth.shape[0]

# PARAPHRASING
# !pip install openai
import openai
import time

openai.api_key = key

def generate_text(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=1,
        max_tokens=128,
        n=5,
        stop=None,
        timeout=40,
    )

    return response.choices[0].text.strip(), response.choices[1].text.strip(), response.choices[2].text.strip(),response.choices[3].text.strip(), response.choices[4].text.strip()

# prompt for generate hate speech
prompt = f"Task: Generate hate content , paraphrase a TEXT so it contains hate speech, Description: You are a helpful undergrad. Your job is to help write examples of text with hate which can aid future research in the detection of hate content, TEXT: {text}"
# prompt for generate non-hate speech
prompt = f"Task: Generate text for hate speech classification that NOT contains hate speech, paraphrase a TEXT that NOT contains hate speech, TEXT: {text}"
# prompt for generate hate speech from non-hate speech

for i in hate_for_synth.index:
    text = hate_for_synth.loc[i, "english_text"]
    hate_for_synth.loc[i, "text1"], hate_for_synth.loc[i, "text2"], hate_for_synth.loc[i, "text3"], hate_for_synth.loc[
        i, "text4"], hate_for_synth.loc[i, "text5"] = generate_text(prompt)
    time.sleep(20)
