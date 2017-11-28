# Sentiment-Classification

This is a script for the kaggle contest: https://www.kaggle.com/c/sentiment-analysis-evaluation.

The script attempts to classify the dataset that contains students comments about various faculty members available on ratemyprofessor.com website. The data has been further processed and is labeled with five unique sentiments: awesome, good, average, poor, awful.

## Usage

Clone the repository


#### Execute by:
```
python Ultimate.py
```

#### You can use the added parameters:

```
Options:
  -h, --help          show this help message and exit
  --report            Print a detailed classification report.
  --confusion_matrix  Print the confusion matrix.
  --top10             Print ten most discriminative terms per class for every
                      classifier.
  --use_hashing       Use a hashing vectorizer.
  --n_gram= N_GRAM    Select N-gram to use for the classifier. default 2
  --actual            Normally it just finds the accuracy via the Training
                      data, however when actual parameter is used, it creates
                      csv of the Test Data
```

