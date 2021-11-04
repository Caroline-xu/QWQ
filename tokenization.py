import pandas as pd
import re
import nltk
from nltk.tokenize import WordPunctTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords
# needed for nltk.pos_tag function nltk.download(’averaged_perceptron_tagger’)
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

#Data Proprocessing
df = pd.read_csv('500_Reddit_users_posts_labels.csv')
pts = df['Post']
pts.to_csv('posts.csv',index=False)
pts.to_csv('posts.txt',index=False,sep='\t')

# the size of the data
l = len(pts)
# tokenization for each comment
clean_token = []
for i in range(l):
    # In each iteration, add an empty list to the main list
    clean_token.append([])

for i in range(l) :
    word_punct_token = WordPunctTokenizer().tokenize(pts[i])
    #removed the tokens which are not a word (normalization)
    k = 0
    for token in word_punct_token:
        token = token.lower()
        # remove any value that are not alphabetical
        new_token = re.sub(r'[^a-zA-Z]+', '', token) 
        # remove empty value and single character value
        if new_token != "" and len(new_token) >= 2: 
            vowels=len([v for v in new_token if v in "aeiou"])
            if vowels != 0: # remove line that only contains consonants
                clean_token[i].append(new_token)