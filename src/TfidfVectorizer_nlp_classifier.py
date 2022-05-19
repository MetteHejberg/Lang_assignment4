# system tools
import os
import sys

# data munging tools
import pandas as pd
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import argparse

# let's load and process the data 
def load_process_data():
    # define path
    filepath = os.path.join("..", "CDS-LANG", "toxic", "VideoCommentsThreatCorpus.csv")
    # read data as pandas data frame
    data = pd.read_csv(filepath)
    # create balanced data
    data_balanced = clf.balance(data, 1000)
    # split the data 
    X = data_balanced["text"]
    y = data_balanced["label"]
    # create train test split
    X_train, X_test, y_train, y_test = train_test_split(X, # texts for the model
                                                            y, # classification labels
                                                            test_size = 0.2, # create an 80/20 split
                                                            random_state = 42) # random state for reproducibility 
    return X_train, X_test, y_train, y_test

# create vectorizer
def vec(X_train, X_test):
    # initialize vectorizer
    vectorizer = TfidfVectorizer(ngram_range = (1,2), # unigrams and bigrams
                                 lowercase = True, # use lower case
                                 max_df = 0.95, # remove very common words 
                                 min_df = 0.05, # remove very rare words 
                                 max_features = 150)
    # fit and transform the data to the vectorizer
    X_train_feats = vectorizer.fit_transform(X_train)
    X_test_feats = vectorizer.transform(X_test)
    return X_train_feats, X_test_feats

# get predictions
def clf_pred(X_train_feats, y_train, X_test_feats, y_test, rep_name):
    # inialize logistic gression classifier
    classifier = LogisticRegression(random_state = 42).fit(X_train_feats, y_train)
    # get predictions
    y_pred = classifier.predict(X_test_feats)
    # get classification report
    clf_report = metrics.classification_report(y_test, y_pred)
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
    # add argparse arguments 
    ap.add_argument("-r", "--rep_name", required=True, help="the name of the classification")
    args = vars(ap.parse_args())
    return args

# let's get the code to run!
def main():
    args = parse_args()
    X_train, X_test, y_train, y_test = load_process_data()
    X_train_feats, X_test_feats = vec(X_train, X_test)
    clf_report = clf_pred(X_train_feats, y_train, X_test_feats, y_test, args["rep_name"])
    
if __name__ == "__main__":
    main()

