from nltk.tokenize import word_tokenize
import pandas as pd
import csv
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from preprocessing import reddit_lemm
from preprocessing import reddit_stem
from preprocessing import stop_words
from preprocessing import df

from sklearn.feature_extraction.text import TfidfVectorizer

label = df.loc[:, "Label"]

#TF-IDF
tfidf = TfidfVectorizer(analyzer=lambda x:[w for w in x if w not in stop_words])
tfidf_vectors = tfidf.fit_transform(reddit_lemm)

df2 = pd.DataFrame(tfidf_vectors.todense(), columns=tfidf.vocabulary_)
df2['label'] = label
print(df2['label'])
compression_opts = dict(method='zip',
                        archive_name='tfidf.csv') 
df2.to_csv('tfidf.zip', index=False,
          compression=compression_opts) 

#CountVectorizer word Frequency
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer=lambda x:[w for w in x if w not in stop_words])
cv_vectors = cv.fit_transform(reddit_lemm)


df3 = pd.DataFrame(cv_vectors.todense(), columns=cv.vocabulary_)
compression_opts = dict(method='zip',
                        archive_name='countvec.csv') 
df3.to_csv('countvec.zip', index=False,
          compression=compression_opts) 


'''''

#Word2Vec
from pyspark.ml.feature import Word2Vec
df = pd.read_csv("reddit_lemm.csv")

word2Vec = Word2Vec(vectorSize=500, minCount=2, inputCol=df, outputCol= "result")
word2vec_m = word2Vec.fit(reddit_lemm)

print(word2vec_m)

'''''
