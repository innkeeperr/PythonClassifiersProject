import nltk as nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn import ensemble
# from keras import layers, models, optimizers
import numpy as np

# import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word

from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QLabel
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import pyqtSlot
import sys


#################################
# OCZYSZCZANIE TEKSTU
# Klasa pomagająca oczyścić tekst znajdujący się w kolumnie DataFrame
class DataFrameCleaner:
    def __init__(self, mainDF, text, score):
        self.mainDF = mainDF
        self.text = text
        self.score = score

    # DataFrame printing
    def dfPrint(self):
        print(self.mainDF)

    # Zmiana liter na małe
    def dfLowerCase(self):
        self.mainDF[self.text] = self.mainDF[self.text].apply(lambda x: " ".join(x.lower() for x in x.split()))
        return (self.mainDF)

    # Usuniecie znaków interpunkcyjnych
    def dfMarksRemove(self):
        self.mainDF[self.text] = self.mainDF[self.text].str.replace('[^\w\s]', '')
        return self.mainDF

    # Usuniecie z opinii "stop words"
    def dfStopwordsRemove(self):
        stop = stopwords.words('english')
        self.mainDF[self.text] = self.mainDF[self.text].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
        return self.mainDF

    # Usunięcie słów, które powtarzają sie najczęściej
    def dfMostFreqWords(self):
        freq = pd.Series(' '.join(self.mainDF[self.text]).split()).value_counts()[:10]
        freq = list(freq.index)
        self.mainDF[self.text] = self.mainDF[self.text].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
        return self.mainDF

    # Usunięcie słów, które powtarzają sie najrzadziej
    def dfLeastOftenWords(self):
        freqMin = pd.Series(' '.join(self.mainDF[self.text]).split()).value_counts()[-10:]
        freqMin = list(freqMin.index)
        self.mainDF[self.text] = self.mainDF[self.text].apply(
            lambda x: " ".join(x for x in x.split() if x not in freqMin))
        return self.mainDF

    # Nieuzywane (zajmuje za dużo czasu
    # Poprawienie słów, ktore zostały napisane z błędem (funkcja correct() wykonuje się bardzo długo)
    # def dfCorrectWords(self):
    #   self.mainDF[self.text].apply(lambda x: str(TextBlob(x).correct()))
    #   return self.mainDF

    # Lematyzacja tekstu (jedynie trzon słów), lematyzacja > stemming
    def dfLemmatize(self):
        self.mainDF[self.text] = self.mainDF[self.text].apply(
            lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        return self.mainDF


#################################
# GUI Creating

class App(QWidget):
    globalAccuracy = ""

    xtrain_tfidf = ""
    train_y = ""
    xvalid_tfidf = ""
    valid_y = ""

    def __init__(self):
        super().__init__()
        self.title = 'Animal Crossing Opinions Prediction'
        self.left = 500
        self.top = 500
        self.width = 700
        self.height = 500
        self.initUI()

        self.df = pd.read_csv('user_reviews.csv', nrows=3000, sep=',')

        self.trainDF = pd.DataFrame()
        self.trainDF["Rating"] = self.df["grade"]
        self.trainDF["Opinion"] = self.df["text"]

        mainTrainDF = DataFrameCleaner(self.trainDF, 'Opinion', 'Rating')

        # Text Normalizing
        self.trainDF = mainTrainDF.dfLowerCase()
        self.trainDF = mainTrainDF.dfMarksRemove()
        self.trainDF = mainTrainDF.dfStopwordsRemove()
        self.trainDF = mainTrainDF.dfMostFreqWords()
        self.trainDF = mainTrainDF.dfLeastOftenWords()
        self.trainDF = mainTrainDF.dfLemmatize()

        print(self.trainDF)

        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(self.trainDF['Opinion'], self.trainDF['Rating'])
        encoder = preprocessing.LabelEncoder()
        self.train_y = encoder.fit_transform(train_y)
        self.valid_y = encoder.fit_transform(valid_y)

        # WEKTORYZACJA TEKSTU
        # Term frequency & inverse document frequency (word level tf-idf)
        self.tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
        self.tfidf_vect.fit(self.trainDF['Opinion'])
        self.xtrain_tfidf = self.tfidf_vect.transform(train_x)
        self.xvalid_tfidf = self.tfidf_vect.transform(valid_x)


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Textbox for input opinions
        # self.textboxInput = QLineEdit(self)
        # self.textboxInput.move(50, 50)
        # self.textboxInput.resize(500, 40)

        start = QPushButton('Linear Classifier', self)
        start.setToolTip('Linear Classifier')
        start.move(100, 100)
        start.clicked.connect(self.on_click_start)

        button1 = QPushButton('Naive Bayes', self)
        button1.setToolTip('Naive Bayes')
        button1.move(250, 100)
        button1.clicked.connect(self.on_click_start2)

        button2 = QPushButton('SVM', self)
        button2.setToolTip('SVM')
        button2.move(400, 100)
        button2.clicked.connect(self.on_click_start3)

        button3 = QPushButton('Random Forest', self)
        button3.setToolTip('Random Forest')
        button3.move(550, 100)
        button3.clicked.connect(self.on_click_start4)

        label = QLabel('LabelAcc', self)
        label.setText("Accuracy")
        label.move(200, 150)

        # Textbox for output
        self.textboxOutput = QLineEdit(self)
        self.textboxOutput.move(200, 165)
        self.textboxOutput.resize(250, 50)

        self.show()

    @pyqtSlot()
    def on_click_start(self):
        # Linear Classifier on Word Level TF IDF Vectors
        classifier = linear_model.LogisticRegression()

        # fit the training dataset on the classifier
        classifier.fit(self.xtrain_tfidf, self.train_y)

        # predict the labels on validation dataset
        predictions = classifier.predict(self.xvalid_tfidf)
        accuracy = metrics.accuracy_score(predictions, self.valid_y)

        print("LR, WordLevel TF-IDF Accuracy: ", accuracy)
        self.globalAccuracy = accuracy
        self.textboxOutput.setText("Accuracy: " + str(accuracy))

    def on_click_start2(self):
      # Naive Bayes on Word Level TF IDF Vectors
      classifier = naive_bayes.MultinomialNB()

      # fit the training dataset on the classifier
      classifier.fit(self.xtrain_tfidf, self.train_y)

      # predict the labels on validation dataset
      predictions = classifier.predict(self.xvalid_tfidf)
      accuracy = metrics.accuracy_score(predictions, self.valid_y)

      print("LR, WordLevel TF-IDF Accuracy: ", accuracy)
      self.globalAccuracy = accuracy
      self.textboxOutput.setText("Accuracy: " + str(accuracy))

    def on_click_start3(self):
      # SVM on Ngram Level TF IDF Vectors
      classifier = svm.SVC()

      # fit the training dataset on the classifier
      classifier.fit(self.xtrain_tfidf, self.train_y)

      # predict the labels on validation dataset
      predictions = classifier.predict(self.xvalid_tfidf)
      accuracy = metrics.accuracy_score(predictions, self.valid_y)

      print("LR, WordLevel TF-IDF Accuracy: ", accuracy)
      self.globalAccuracy = accuracy
      self.textboxOutput.setText("Accuracy: " + str(accuracy))

    def on_click_start4(self):
      # RF on Word Level TF IDF Vectors
      classifier = ensemble.RandomForestClassifier()

      # fit the training dataset on the classifier
      classifier.fit(self.xtrain_tfidf, self.train_y)

      # predict the labels on validation dataset
      predictions = classifier.predict(self.xvalid_tfidf)
      accuracy = metrics.accuracy_score(predictions, self.valid_y)

      print("LR, WordLevel TF-IDF Accuracy: ", accuracy)
      self.globalAccuracy = accuracy
      self.textboxOutput.setText("Accuracy: " + str(accuracy))


if __name__ == '__main__':
    display = pd.options.display
    display.width = None
    display.max_columns = 50
    display.max_rows = 50
    display.max_colwidth = 199

    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())

#################################

# Klasa służąca do opisu DataFrame'u, tworzy nowe kolumny do DF pokazujące różne parametry tesktu
class TextDescriber:
    def __init__(self, newDataFrame, text, score):
        self.newDataFrame = newDataFrame
        self.text = text
        self.score = score

    # DataFrame printing
    def dfPrint(self):
        print(self.newDataFrame)

    # DataFrame describing
    def dfDescribe(self):
        print(self.newDataFrame.describe())

    # wyliczenie ilości słów w opinii
    def wordCounter(self):
        self.newDataFrame['word_count'] = self.newDataFrame[self.text].apply(lambda x: len(str(x).split(" ")))
        print(self.newDataFrame)

    # Wyliczenie ilości znaków w opinii
    def charCounter(self):
        self.newDataFrame['char_count'] = self.newDataFrame[self.text].str.len()
        print(self.newDataFrame)

    # Średnia długość słowa
    def avgWord(self):
        self.newDataFrame['avg_word'] = self.newDataFrame[self.text].apply(
            lambda x: sum(len(x) for x in x.split()) / len(x.split()))
        print(self.newDataFrame)

    # Zapisanie stop words w osobnej kolumnie (liczba stopwords)
    def stopwordsCounter(self):
        stop = stopwords.words('english')
        self.newDataFrame['stopwords'] = self.newDataFrame[self.text].apply(
            lambda x: len([x for x in x.split() if x in stop]))

    # Ile liczb w tekście (do usuniecia)
    def numericCounter(self):
        self.newDataFrame['numerics'] = self.newDataFrame[self.text].apply(
            lambda x: len([x for x in x.split() if x.isdigit()]))


