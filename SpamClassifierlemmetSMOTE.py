# -*- coding: latin-1 -*-
"""
Created on Wed Jul  8 20:34:17 2020

@author: DELL
"""

import nltk
import pandas as pd
import numpy as np

data=pd.read_csv('spam.csv', sep=',', encoding='latin-1')
data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1, inplace=True)

#Data cleaning and preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

lemmet=WordNetLemmatizer()
corpus=[]

for i in range(len(data)):
    words=re.sub('[^a-zA-Z]', ' ', data['Message'][i])
    words=words.lower()
    words=words.split()
    words=[lemmet.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    words=' '.join(words)
    corpus.append(words)

y=pd.get_dummies(data['Type'])
y=y.iloc[:,1].values

#creating BagOfWords
from sklearn.feature_extraction.text import TfidfVectorizer
TFIDF=TfidfVectorizer()
x=TFIDF.fit_transform(corpus).toarray()

#Model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y, test_size=.2, random_state=1)

#Spam count vs ham count ration differes
print('lenght of dependent variable', len(y_train))
print('lenght of ham in dependent variable', len(y_train[y_train==0]))
print('lenght of spam in dependent variable', len(y_train[y_train==1]))

#Smote technique used
from imblearn.over_sampling import SMOTE
os=SMOTE()
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
#columns=X_train.columns
os_data_X = pd.DataFrame(data=os_data_X )
os_data_y= pd.DataFrame(data=os_data_y)


#Model
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
mnb=MultinomialNB().fit(os_data_X,os_data_y)
bnb=BernoulliNB().fit(os_data_X,os_data_y)

mnb_prediction=mnb.predict(X_test)
bnb_prediction=bnb.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
mnb_confution=confusion_matrix(y_test,mnb_prediction)
bnb_confution=confusion_matrix(y_test,bnb_prediction)
mnb_accuracy=accuracy_score(y_test,mnb_prediction)
bnb_accuracy=accuracy_score(y_test,bnb_prediction)


print('BernoulliNB accuracy ',bnb_accuracy)
print('MultinomialNB accuray', mnb_accuracy)




