# import packages
import os
import sys

# tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, 
                                    Flatten,
                                    Conv1D, 
                                    MaxPooling1D, 
                                    Embedding)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

# scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, 
                            classification_report)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

# other packages
import argparse
import pandas as pd
import utils.classifier_utils as clf
import re
import tqdm
import unicodedata
import contractions
from bs4 import BeautifulSoup  # Good for working with .html
import nltk                    # package for tokenization 
nltk.download('punkt')

# initial function definitions
# cleaning up the text
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

# cleaning up the text
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

# preprocessing function
def pre_process_corpus(docs):
    norm_docs = []
    for doc in tqdm.tqdm(docs):
        doc = strip_html_tags(doc)
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))
        doc = doc.lower()
        doc = remove_accented_chars(doc)
        doc = contractions.fix(doc)
        # lower case and remove special characters\whitespaces
        doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()  
        norm_docs.append(doc)
    return norm_docs

# let's load the data 
def load_process_data():
    filepath = os.path.join("in", "toxic", "VideoCommentsThreatCorpus.csv")
    data = pd.read_csv(filepath)
    # balance the data set
    data_balanced = clf.balance(data, 1000)
    X = data_balanced["text"]
    y = data_balanced["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, # texts for the model
                                                            y, # classification labels
                                                            test_size = 0.2, # create an 80/20 split
                                                            random_state = 42) # random state for reproducibility 
    # prepocessing 
    X_train_norm = pre_process_corpus(X_train) 
    X_test_norm = pre_process_corpus(X_test)
    # initalize tokenizer
    t = Tokenizer(oov_token = '<UNK>')
    # fit the data to the tokenizer
    t.fit_on_texts(X_train_norm)
    t.word_index["<PAD>"] = 0
    X_train_seqs = t.texts_to_sequences(X_train_norm)
    X_test_seqs = t.texts_to_sequences(X_test_norm)
    MAX_SEQUENCE_LENGTH = 1000
    X_train_pad = sequence.pad_sequences(X_train_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
    X_test_pad = sequence.pad_sequences(X_test_seqs, maxlen=MAX_SEQUENCE_LENGTH)
    return X_train_pad, X_test_pad, y_train, y_test, t, MAX_SEQUENCE_LENGTH

# create the model
def mdl(y_train, y_test, t, eps, b_size, MAX_SEQUENCE_LENGTH):
    # label encoder
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    # set parameters
    VOCAB_SIZE = len(t.word_index)
    EMBED_SIZE = 300
    EPOCHS = eps
    BATCH_SIZE = b_size
    # initialize model
    model = Sequential()
    # add layers
    model.add(Embedding(VOCAB_SIZE, 
                    EMBED_SIZE, 
                    input_length=MAX_SEQUENCE_LENGTH))
    model.add(Conv1D(filters=128, 
                        kernel_size=4, 
                        padding='same',
                        activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, 
                        kernel_size=4, 
                        padding='same', 
                        activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, 
                        kernel_size=4, 
                        padding='same', 
                        activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                        optimizer='sgd', 
                        metrics=['accuracy'])
    return model, EPOCHS, BATCH_SIZE

# train model
def train(model, X_train_pad, y_train, EPOCHS, BATCH_SIZE, X_test_pad, y_test, rep_name):
    # fit data to the model
    model.fit(X_train_pad, y_train,
                    epochs = EPOCHS,
                    batch_size = BATCH_SIZE,
                    validation_split = 0.1,
                    verbose = True)
    # get predictions
    predictions = (model.predict(X_test_pad) > 0.5).astype("int32")
    # get classification report
    clf_report = classification_report(y_test, predictions)
    # print report
    print(clf_report)
    # define outpath
    p = os.path.join("out", rep_name)
    # save classification report
    sys.stdout = open(p, "w")
    text_file = print(clf_report)

def parse_args():
    # initialize argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--rep_name", required=True, help="the name of the classification report")
    ap.add_argument("-e", "--eps", type=int, required=True, help="number of epochs of the model")
    ap.add_argument("-b", "--b_size", type=int, required=True, help="the batch size of the model")
    args = vars(ap.parse_args())
    return args

# let's run the code
def main():
    args = parse_args()
    X_train_pad, X_test_pad, y_train, y_test, t, MAX_SEQUENCE_LENGTH = load_process_data()
    model, EPOCHS, BATCH_SIZE = mdl(y_train, y_test, t, args["eps"], args["b_size"], MAX_SEQUENCE_LENGTH)
    report_df = train(model, X_train_pad, y_train, args["eps"], args["b_size"], X_test_pad, y_test, args["rep_name"])
    
if __name__ == "__main__":
    main()
    
