
import pandas as pd
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import WordPunctTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

#Data Proprocessing
df = pd.read_csv('500_Reddit_users_posts_labels.csv')

#define stop words
stop_words = list(set(stopwords.words('english')))

#tokenize "Post" column, can create new column called "tokenized_post"
df['Tokenized Post'] = df.apply(lambda row: nltk.word_tokenize(row['Post']), axis=1)

#iterrate over "tokenized_posts" cloumn
for index, value in df["Tokenized Post"].items():
    #contain the clean tokens of a post, set list to empty every time finish iterating a post
    clean_token=[]
    #for each word in a single post
    for token in value:
        #remove any value that are not alphabetical
        new_token = re.sub(r'[^a-zA-Z]+', '', token.lower()) 
        #remove empty value and single character value, remove stop words
        if (new_token != "") and (len(new_token) >= 2) and (new_token not in stop_words): 
            vowels=len([v for v in new_token if v in "aeiou"])
            if vowels != 0: #remove line that only contains consonants
                clean_token.append(new_token)
    #change data in "Tokenized Post" to list of tokenized words with stop words removed 
    df["Tokenized Post"][index]=clean_token

'''  
To check if there is stop word:
num=0
for i in df["Tokenized Post"]:
    for j in i:
        if j in stop_words:
            num+=1
print(num)
'''

#for testing
print(df.head(4))

df.to_csv("clean_posts.csv",index=False)