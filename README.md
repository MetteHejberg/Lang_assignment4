## 1. Assignment 4: Text classification
The assignment for this week builds on these concepts and techniques. We're going to be working with the data in the folder ```CDS-LANG/toxic``` and trying to see if we can predict whether or not a comment is a certain kind of *toxic speech*. You should write two scripts which do the following:

- The first script should perform benchmark classification using standard machine learning approaches
  - This means ```CountVectorizer()``` or ```TfidfVectorizer()```, ```LogisticRegression``` classifier
  - Save the results from the classification report to a text file
- The second script should perform classification using the kind of deep learning methods we saw in class
  - Keras ```Embedding``` layer, Convolutional Neural Network
  - Save the classification report to a text file 

## 2. Methods

Labels in the classification report:
- Non-threat = 0
- Threat = 1

## 3.1 Usage ```CNN_nlp_classifier.py```
To run the code you should:
- Pull this repository with this folder structure 
- Place the data in ```in```
- Install the packages mentioned in ```requirements.txt```
- Make sure the ```utils``` folder is placed inside the ```src``` folder
- Set your current working directory to the level about ```src```
- To run the code, write in the command line: ```python src/cnn_nlp_classifier.py -e "number of epochs of the model -b "the batch size of the model" -r "cnn_classification_report.txt"```
  - I wrote the following in the command line to produce the results in ```out```: ```python src/cnn_nlp_classifier.py -e 20 -b 32 -r "cnn_classification_report.txt"```

## 3.2 Usage ```TfidfVectorizer_nlp_classifier.py```
To run the code you should:
- Pull this repository with this folder structure
- Place the data in ```in```
- Install the packages mentioned in ```requirements.txt```
- Make sure the ```utils``` folder is placed inside the ```src``` folder
- Set your current working directory to the level about ```src```
- To run run the code, write in the command line: ```python src/CNN_nlp_classifier.py -r "report name"```
  - I wrote the following in the command line to produce the results in ```out```: ```python src/TfidfVectorizer_nlp_classifier.py -r "tfidf_classification_report.txt"``` 

## 4. Discussion of results 
The Tfidf vectorizer and logistic regression classifier performs significantly better on this classification task than the convolutional neural network. the cnn classifier only manages to learning something on the non-threat texts, resulting in an overall accuracy on the entire task of 50% which is chance level. The Tfidf vectorizer and logistic regression classifier ends with an overall accuracy of 76% with only a slightly better accuracy on the non-threat texts than the threat texts. 


