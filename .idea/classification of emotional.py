# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

df = pd.read_csv('F:\machineLearning\movie_data.csv')

# 关于数据源预处理部分还是有问题，主要集中在正则表达式部分有问题
def prepocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emotioncons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text = re.sub('[\w]+',' ',text.lower()) + ''.join(emotioncons).replace('-','')
    return text

tem = df['review'] = df['review'].apply(prepocessor)
print (tem)



