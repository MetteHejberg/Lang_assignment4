## 1. Assignment 4: Text classification
Link to repository: https://github.com/MetteHejberg/Lang_assignment4

The assignment for this week builds on these concepts and techniques. We're going to be working with the data in the folder ```CDS-LANG/toxic``` and trying to see if we can predict whether or not a comment is a certain kind of *toxic speech*. You should write two scripts which do the following:

- The first script should perform benchmark classification using standard machine learning approaches
  - This means ```CountVectorizer()``` or ```TfidfVectorizer()```, ```LogisticRegression``` classifier
  - Save the results from the classification report to a text file
- The second script should perform classification using the kind of deep learning methods we saw in class
  - Keras ```Embedding``` layer, Convolutional Neural Network
  - Save the classification report to a text file 

## 2. Methods
This repository contains two scripts that performs classification of threat vs. non-threat language in posts from posts on the internet. 

One script uses a convolutional neural network with ```keras``` to classify the text. The user can set the number of epochs of the model and the batch size of the model. The classification report is saved to ```out```.

This approach uses deep learning, which can, with enough data, yield accurate results eventough is knows nothing about language.

The second script uses the ```tfidf vectorizer``` to create a numeric representation of the text, which is then fed through a logistic regression classifier. The classification report is also saved to ```out```

The vectorizer assume the bag of words notion of language. Language can be numerically represented however some information is lost such as positional information - we only keep the frequency of the words nothing about how the words relate to other words. However, it is a fast and easy way to create a numerical representation of a document which can yield very accurate results. 

Labels in the classification reports:
- Non-threat = 0
- Threat = 1

Get the data here: https://www.simula.no/sites/default/files/publications/files/cbmi2019_youtube_threat_corpus.pdf

## 3.1 Usage ```CNN_nlp_classifier.py```
To run the code you should:
- Pull this repository with this folder structure 
- Place the data in a folder called ```toxic``` in ```in```
- Install the packages mentioned in ```requirements.txt```
- Make sure the ```utils``` folder is placed inside the ```src``` folder
- Set your current working directory to the level about ```src```
- To run the code, write in the command line: ```python src/cnn_nlp_classifier.py -e "number of epochs of the model -b "the batch size of the model" -r "cnn_classification_report.txt"```
  - I wrote the following in the command line to produce the results in ```out```: ```python src/cnn_nlp_classifier.py -e 20 -b 32 -r "cnn_classification_report.txt"```

## 3.2 Usage ```TfidfVectorizer_nlp_classifier.py```
To run the code you should:
- Pull this repository with this folder structure
- Place the data in a folder called ```toxic``` in ```in```
- Install the packages mentioned in ```requirements.txt```
- Make sure the ```utils``` folder is placed inside the ```src``` folder
- Set your current working directory to the level about ```src```
- To run run the code, write in the command line: ```python src/CNN_nlp_classifier.py -r "report name"```
  - I wrote the following in the command line to produce the results in ```out```: ```python src/TfidfVectorizer_nlp_classifier.py -r "tfidf_classification_report.txt"``` 

## 4. Discussion of results 
The Tfidf vectorizer and logistic regression classifier performs significantly better on this classification task than the convolutional neural network. the cnn classifier only manages to learning something on the non-threat texts, resulting in an overall accuracy on the entire task of 50% which is chance level. The Tfidf vectorizer and logistic regression classifier ends with an overall accuracy of 76% with only a slightly better accuracy on the non-threat texts than the threat texts. Perhaps the cnn model would perform better if it had more data.

