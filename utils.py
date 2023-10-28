#import...


regexp = RegexpTokenizer('\w+')
!python -m spacy download el_core_news_lg

def lemmatize(text):
    nlp=spacy.load("el_core_news_lg")
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

def preprocess_text(text):
    stop= pd.read_csv("stopwords__greek.csv", header=None )
    stop= stop.values.tolist()
    stopw = [item for sublist in stop for item in sublist]
    stopwords=set(stopw)
    # Tokenize the text using regular expressions
    tokenizer = RegexpTokenizer('\w+')
    tokens = tokenizer.tokenize(text)
    # Lemmatize the tokens using spaCy
    lemmatized_tokens = [token.lemma_ for token in nlp(" ".join(tokens))]
    # Combine the lemmatized tokens into a single string
    processed_text = " ".join(lemmatized_tokens)
    # Clean accent characters
    processed_text = clean_accent(processed_text)
    # Convert to lowercase
    processed_text = processed_text.lower()
    # Remove links
    processed_text = re.sub(r'https?:\/\/.*[\r\n]*', '', processed_text)
    # Remove hashtags
    processed_text = re.sub(r'#', '', processed_text)
    # Remove names (mentions)
    processed_text = re.sub(r'@[A-Za-z0-9_]+', '', processed_text)
    # Remove punctuation
    processed_text = re.sub(r'[^\w\s]', '', processed_text)
    # Remove numbers
    processed_text = drop_numbers(processed_text)
    # Tokenize the processed text
    tokens = tokenizer.tokenize(processed_text)
    # Remove stopwords and keep words with more than 2 characters
    tokens = [token for token in tokens if token not in stopwords and len(token) > 2]
    # Combine tokens into a single string
    processed_text = " ".join(tokens)
    return processed_text
