# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:18:02 2019

@author: pltru
"""

#load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from collections import Counter
import os


reviews = pd.read_csv("Documents\\reviews_cleaned.csv")

print('Number of missing comments in comment text:')
reviews['comments'].isnull().sum()
reviews = reviews.dropna()

# EDA
# Calculating number of comments in each category

table1 = reviews.pivot_table(index = 'ratings', 
         values = 'comments', aggfunc = 'count')

table1 = table1.reset_index()

plt.figure(figsize=(5,5))

ax= sns.barplot(x= 'ratings', y = 'comments', data = table1)

plt.title("Comments in each rating", fontsize=24)
plt.ylabel('Number of comments', fontsize=18)
plt.xlabel('Rating ', fontsize=18)

#adding the text labels
rects = ax.patches
labels = table1.comments
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=10)

plt.show()


#distribution of number of words in comments text
lens = reviews.comments.str.len()
lens.hist(bins = np.arange(0,3000,50))
plt.title("Distribution of number of words in reviews")
plt.ylabel("Count")
plt.xlabel("Number of characters")


#word cloud for most words used in each category of rating

from wordcloud import WordCloud,STOPWORDS

plt.figure(figsize=(40,25))

# 1 star
subset = reviews[reviews.ratings==1]
text = subset.comments.values
cloud_1 = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

plt.subplot(2, 3, 1)
plt.axis('off')
plt.title("1 stars rating",fontsize=30)
plt.imshow(cloud_1)


# 2 star
subset = reviews[reviews.ratings==2]
text = subset.comments.values
cloud_2 = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

plt.subplot(2, 3, 2)
plt.axis('off')
plt.title("2 stars rating",fontsize=30)
plt.imshow(cloud_2)

#3 star
subset = reviews[reviews.ratings==3]
text = subset.comments.values
cloud_3 = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))
plt.subplot(2, 3, 3)
plt.axis('off')
plt.title("3 stars rating",fontsize=30)
plt.imshow(cloud_3)


#4 star
subset = reviews[reviews.ratings==4]
text = subset.comments.values
cloud_4 = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

plt.subplot(2, 3, 4)
plt.axis('off')
plt.title("4 stars rating",fontsize=30)
plt.imshow(cloud_4)

#5 star
subset = reviews[reviews.ratings==5]
text = subset.comments.values
cloud_5 = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

plt.subplot(2, 3, 5)
plt.axis('off')
plt.title("5 stars rating",fontsize=30)
plt.imshow(cloud_5)

#data pre-processing

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import re
import sys
import warnings
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
nltk.download('stopwords')


stemmer = SnowballStemmer('english')
words = stopwords.words("english")

reviews['cleaned'] = reviews['comments'].apply(lambda x: " " .join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]"," ", x).split() if i not in words]).lower())

#data train/test

X_train, X_test, y_train, y_test = train_test_split(reviews['cleaned'], reviews.ratings, test_size = 0.3, shuffle = True)

#Linear SVC model

pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1,3), stop_words = "english", sublinear_tf = True, norm = 'l2', analyzer = 'word')),
        ('chi', SelectKBest(chi2, k = 10000)),
        ('clf', LinearSVC(C=1.0, penalty='l1', max_iter = 3000, dual = False))])
    
SVC_model = pipeline.fit(X_train, y_train)

vectorizer = SVC_model.named_steps['vect']
chi = SVC_model.named_steps['chi']
clf = SVC_model.named_steps['clf']

print(clf)

#key words + accuracy rates

feature_names = vectorizer.get_feature_names()
feature_names = [feature_names[i] for i in chi.get_support(indices = True)]
feature_names = np.asarray(feature_names)

target_names = ['1','2','3','4','5']
print("top 15 keywords per class:")
for i, label in enumerate(target_names):
    top15 = np.argsort(clf.coef_[i])[-15:]
    print("%s: %s" % (label, " ".join(feature_names[top15])))

print("accuracy score:" + str(SVC_model.score(X_test, y_test)))
print("f1 score:" + str(f1_score(y_test, SVC_model.predict(X_test), average = "weighted")))


#example

print(SVC_model.predict(['Very bad service and experience!']))
print(SVC_model.predict(['This was an average place to stay, we had moderate amount of fun']))
print(SVC_model.predict(['Beautiful place and apartment']))



#confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(SVC_model.predict(X_test),y_test)
cm= pd.DataFrame(np.reshape((0,0,0,0,0,0,0,0,0,1,10,25,3846,1579,823,0,14,1940,5247,2169,1,31,5534,11663,26868),(5,5)), 
             columns = column_names, index = row_names)

column_names = ['1', '2', '3','4','5']
row_names    = ['1', '2', '3','4','5']



sns.heatmap(cm, annot = True, linewidths = 0.5,linecolor = "red",fmt = "g")
plt.title("Test for Test Dataset")
plt.xlabel("real y values")
plt.ylabel("predicted y values")
plt.show()



#Naive Baise
 
NB_pipeline  = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words = "english", ngram_range=(1,3))),
                ('clf', OneVsRestClassifier(MultinomialNB(
                    fit_prior=True, class_prior=None))),
            ])
                
NB_model = NB_pipeline.fit(X_train, y_train)


NB_vectorizer = NB_model.named_steps['tfidf']
NB_chi = NB_model.named_steps['chi']
NB_clf = NB_model.named_steps['clf']

#key words + accuracy rates

feature_names = NB_vectorizer.get_feature_names()
feature_names = [feature_names[i] for i in NB_chi.get_support(indices = True)]
feature_names = np.asarray(feature_names)

target_names = ['1','2','3','4','5']
print("top 15 keywords per class:")
for i, label in enumerate(target_names):
    top15 = np.argsort(NB_clf.coef_[i])[-15:]
    print("%s: %s" % (label, " ".join(feature_names[top15])))
    
print("accuracy score:" + str(NB_model.score(X_test, y_test)))
print("f1 score:" + str(f1_score(y_test, NB_model.predict(X_test), average = "weighted")))

#confusion matrix

cm = confusion_matrix(NB_model.predict(X_test),y_test)
cm= pd.DataFrame(np.reshape((0,0,0,0,0,0,0,0,0,0,1,1,605,42,16,1,1,276,1292,67,9,68,10439,17155,29778),(5,5)), 
             columns = column_names, index = row_names)


sns.heatmap(cm, annot = True, linewidths = 0.5,linecolor = "red",fmt = "g")
plt.title("Test for Test Dataset")
plt.xlabel("real y values")
plt.ylabel("predicted y values")
plt.show()

#Logistic Regression

LogReg_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1,3), stop_words = "english")),
                ('chi', SelectKBest(chi2, k = 10000)),
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
            ])

LogReg_model = LogReg_pipeline.fit(X_train, y_train)

LogReg_vectorizer = LogReg_model.named_steps['tfidf']
LogReg_chi = LogReg_model.named_steps['chi']
LogReg_clf = LogReg_model.named_steps['clf']

#key words + accuracy rates

feature_names = LogReg_vectorizer.get_feature_names()
feature_names = [feature_names[i] for i in LogReg_chi.get_support(indices = True)]
feature_names = np.asarray(feature_names)

target_names = ['1','2','3','4','5']
print("top 15 keywords per class:")
for i, label in enumerate(target_names):
    top15 = np.argsort(LogReg_clf.coef_[i])[-15:]
    print("%s: %s" % (label, " ".join(feature_names[top15])))
    
print("accuracy score:" + str(LogReg_model.score(X_test, y_test)))
print("f1 score:" + str(f1_score(y_test, LogReg_model.predict(X_test), average = "weighted")))

#confusion matrix

cm = confusion_matrix(LogReg_model.predict(X_test),y_test)
cm= pd.DataFrame(np.reshape((0,0,0,0,0,0,0,0,0,0,9,20,3194,1356,687,0,12,1936,4749,1970,2,38,6190,12384,27204),(5,5)), 
             columns = column_names, index = row_names)


sns.heatmap(cm, annot = True, linewidths = 0.5,linecolor = "red",fmt = "g")
plt.title("Test for Test Dataset")
plt.xlabel("real y values")
plt.ylabel("predicted y values")
plt.show()


