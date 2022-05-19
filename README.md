## 1. Assignment 4: Text classification
The assignment for this week builds on these concepts and techniques. We're going to be working with the data in the folder ```CDS-LANG/toxic``` and trying to see if we can predict whether or not a comment is a certain kind of *toxic speech*. You should write two scripts which do the following:

- The first script should perform benchmark classification using standard machine learning approaches
  - This means ```CountVectorizer()``` or ```TfidfVectorizer()```, ```LogisticRegression``` classifier
  - Save the results from the classification report to a text file
- The second script should perform classification using the kind of deep learning methods we saw in class
  - Keras ```Embedding``` layer, Convolutional Neural Network
  - Save the classification report to a text file 

## 2. Methods

Non-threat = 0
Threat = 1

## 3.1 Usage ```CNN_nlp_classifier.py```
To run the code you should:
- Pull this repository with this folder structure 
- Place the data in ```in```
- Install the packages mentioned in ```requirements.txt```
- Make sure the ```utils``` folder is placed inside the ```src``` folder
- Set your current working directory to the level about ```src```
- To run the code, write in the command line: ```python src/CNN_nlp_classifier.py -r "report name" + argparse``` 
  - what I wrote in the command line to produce the results 

## 3.2 Usage ```TfidfVectorizer_nlp_classifier.py```
To run the code you should:
- Pull this repository with this folder structure
- Place the data in ```in```
- Install the packages mentioned in ```requirements.txt```
- Make sure the ```utils``` folder is placed inside the ```src``` folder
- Set your current working directory to the level about ```src```
- To run run the code, write in the command line: ```python src/CNN_nlp_classifier.py -r "report name"```

## 4. Discussion of results 
