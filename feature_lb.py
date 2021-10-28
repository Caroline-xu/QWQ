import pandas as pd
import csv
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from preprocessing import reddit_lemm
from preprocessing import reddit_stem
from preprocessing import stop_words

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(analyzer=lambda x:[w for w in x if w not in stop_words])
tfidf_vectors = tfidf.fit_transform(reddit_lemm)

df = pd.DataFrame(tfidf_vectors.todense(), columns=tfidf.vocabulary_)
print(df)
compression_opts = dict(method='zip',
                        archive_name='tfidf.csv') 
df.to_csv('tfidf.zip', index=False,
          compression=compression_opts) 

