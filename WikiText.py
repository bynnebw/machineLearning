# -*- coding: utf-8 -*-

'''
使用聚类和相似度模型，进行维基百科人物相似度分析
'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

people = pd.read_csv('F:\machineLearning\wiki_people.csv')
obama = people[people['name']=='Barack Obama']


# 使用sklearn来进行词袋模型的建模
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(obama['text'])

word = vectorizer.get_feature_names()
word = np.array(word)
obama_word_count_table = pd.DataFrame({'word': word, 'count': X.toarray()[0]})
obama_word_count_table.sort_values(["count"], ascending=False)

# 使用Tf idf表示和相似度计算
transformer = TfidfTransformer(smooth_idf=False)
X = vectorizer.fit_transform(people['text'])
clinton = people[people['name'] == 'Bill Clinton']

tfidf_clinton = transformer.fit_transform(X, X[36452])

beckham = people[people['name'] == 'David Beckham']
tfidf_beckham = transformer.fit_transform(X, X[23386])

con_similarity = cosine_similarity(tfidf_clinton, tfidf_beckham)
print (con_similarity)

# 会出现内存异常的错误