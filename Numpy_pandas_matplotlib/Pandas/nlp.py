# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:42:25 2020

@author: cdragon-ljl
"""


import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
import fasttext

train_df = pd.read_csv('D:/Dataset/nlp/train_set.csv', sep='\t', nrows=15000)
print(train_df)

print("*"*50)

train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
print(train_df['text_len'].describe())

print("*"*50)

plt.hist(train_df['text_len'], bins=200)
plt.xlabel('Text char count')
plt.title("Histogram of char count")

plt.show()   

train_df['label'].value_counts().plot(kind='bar')
plt.title('News class count')
plt.xlabel("category")

plt.show()   

all_lines = ' '.join(list(train_df['text']))                 
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:d[1], reverse=True)

print(len(word_count))
print(word_count[0])
print(word_count[-1])

print("*"*50)

train_df['text_unique'] = train_df['text'].apply(lambda x: ' '.join(list(set(x.split(' ')))))
all_lines = ' '.join(list(train_df['text_unique']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:int(d[1]), reverse=True)

print(word_count[0])
print(word_count[1])
print(word_count[2])

print("*"*50)


# Count Vectors + RidgeClassifier
# vectorizer = CountVectorizer(max_features=3000)
# train_test = vectorizer.fit_transform(train_df['text'])

# clf = RidgeClassifier()
# clf.fit(train_test[:10000], train_df['label'].values[:10000])

# val_pred = clf.predict(train_test[10000:])
# print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))

#TF-IDF + RidgeClassifier
# tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
# train_test = tfidf.fit_transform(train_df['text'])

# clf = RidgeClassifier()
# clf.fit(train_test[:10000], train_df['label'].values[:10000])

# val_pred = clf.predict(train_test[10000:])
# print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))

train_df['label_ft'] = '__label__' + train_df['label'].astype(str)
train_df[['text', 'label_ft']].iloc[:-5000].to_csv('train.csv', index=None, header=None, sep='\t')

model = fasttext.train_supervised('train.csv', lr=1.0, wordNgrams=2, verbose=2, minCount=1, epoch=25, loss="hs")

val_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[-5000:]['text']]
print(f1_score(train_df['label'].values[-5000:].astype(str), val_pred, average='macro'))