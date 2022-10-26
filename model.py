import pandas as pd
import numpy as np
import string
import re
import string
import contractions
from cleantext import clean
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

def model(msg):    
    def count_punc(mystr):
        return len([c for c in mystr if c in string.punctuation])

    df = pd.read_csv('datasets/SMSSpamCollection', sep='\t', header=None, names=['label', 'text'])
    df['length'] = df['text'].apply(len)
    df['punc'] = df['text'].apply(lambda x: count_punc(x))

    df.to_csv('datasets/sms1.csv', index=False)
    df = pd.read_csv('datasets/sms1.csv')



    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    wn = WordNetLemmatizer()

    def text_preprocessing(mystr):
        mystr = mystr.lower()                                               # case folding
        mystr = re.sub('\w*\d\w*', '', mystr)                               # remove digits
        mystr = re.sub('\n', ' ', mystr)                                    # replace new line characters with space
        mystr = re.sub('[‘’“”…]', '', mystr)                                # removing double quotes and single quotes
        mystr = re.sub('<.*?>', '', mystr)                                  # removing html tags 
        mystr = re.sub(r'\[.*?\]', '', mystr)                               # remove text in square brackets
        mystr = re.sub('https?://\S+|www.\.\S+', '', mystr)                 # removing URLs
        mystr = re.sub('\n', ' ', mystr)                                    # replace new line characters with space
        # mystr = clean(mystr, no_emoji=True)                                 # remove emojis
        mystr = ''.join([c for c in mystr if c not in string.punctuation])  # remove punctuations
        mystr = ' '.join([contractions.fix(word) for word in mystr.split()])# expand contractions
        
        tokens = word_tokenize(mystr)                                       # tokenize the string
        mystr = ''.join([c for c in mystr if c not in string.punctuation])  # remove punctuations
        tokens = [token for token in tokens if token not in stop_words]     # remove stopwords
    #   tokens = [ps.stem(token) for token in tokens]                       # stemming
        tokens = [wn.lemmatize(token) for token in tokens]                   # lemmatization
        new_str = ' '.join(tokens)
        return new_str

    df['processed_text'] = df['text'].apply(lambda x: text_preprocessing(x))
    cv = CountVectorizer() 
    bow = cv.fit_transform(df['processed_text']) 

    total_cells = bow.shape[0] * bow.shape[1]
    nonzero_cells = bow.nnz
    percentage = (nonzero_cells/total_cells)*100
    dtm = pd.DataFrame(data=bow.todense())

    X = bow
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5, shuffle=True)
    model = MultinomialNB()

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

    df = pd.read_csv("datasets/sms1.csv")
    X=df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

    
    pipe = Pipeline([ ('bow'  , CountVectorizer(preprocessor=text_preprocessing)),
                    ('model' , MultinomialNB())
                    ])

    pipe.fit(X=X_train, y=y_train)
    new_sms = [msg]
    return pipe.predict(new_sms)                


e = model("tttttt")    
print(e)