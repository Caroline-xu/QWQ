import pandas as pd
import re
import nltk

#Data Proprocessing
df = pd.read_csv('500_Reddit_users_posts_labels.csv')
pts = df['Post']
pts.to_csv('posts.csv',index=False)
pts.to_csv('posts.txt',index=False,sep='\t')
#split it into words
tokens = nltk.word_tokenize(pts)




