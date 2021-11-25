import pandas as pd
import re
import nltk
import csv
#nltk.download('punkt')
from nltk.tokenize import WordPunctTokenizer
#nltk.download('stopwords')
from nltk.corpus import stopwords
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import os

def tokenized(df):

    #define stop words
    stop_words = list(set(stopwords.words('english')))

    #tokenize "Post" column, can create new column called "tokenized_post"
    df['Tokenized Post'] = df.apply(lambda row: nltk.word_tokenize(row['Post']), axis=1)
    # add new column called "New Label" which convert labels to number from 0-4
    df['New Label'] = df['Label'].replace(['Supportive','Indicator','Ideation','Behavior','Attempt'],[0, 1, 2, 3, 4])

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
    return df

################################ Lematization & Stemming ##########################################
##lemmatization
def lemmAndStem(df):
    lemmatizer = WordNetLemmatizer()
    ##stemming
    stemmer = PorterStemmer()
    #create a lemmatization list for reddit posts
    reddit_lemm = []
    #create a stemming list for reddit posts
    reddit_stem = []

    df = tokenized(df)

    ######need to change this hardcode(500) to the length of post !!!!!!
    for i in range(500):
        #initilize single post list
        singlePost_stem = []
        singlePost_lemm = []
        #read file line by line
        line = df["Tokenized Post"][i]
        #read each line word by word
        for w in line:
            #get one token of (w)th post in stemming method
            rootWord_stem = stemmer.stem(w) 
            #get one token of (w)th post in lemmatization method
            rootWord_lemm = lemmatizer.lemmatize(w)
            #add word after stemming method as an element to single post list
            singlePost_stem.append(rootWord_stem)
            #add word after lemmatization method as an element to single post list
            singlePost_lemm.append(rootWord_lemm)
        #after quit w loop, we could get one post in stemming method, then add it to a large list of all posts
        reddit_stem.append(singlePost_stem)
        #after quit w loop, we could get one post in lemming method, then add it to a large list of all posts
        reddit_lemm.append(singlePost_lemm)

    #"create csv file for stemming posts
    # (and don't use Mac's [Numbers] to open csv file because numbers only could display 1000 columns,
    # but this file contains 7000+ columns, use excel to open it could avoid data loss!)
    file_stem = open('reddit_stem.csv','w+',newline = '')
    with file_stem:
        write_stem = csv.writer(file_stem)
        write_stem.writerows(reddit_stem)
        
    #create csv file for lemmatization posts
    file_lemm = open('reddit_lemm.csv','w+',newline = '')
    with file_lemm:
        write_lemm = csv.writer(file_lemm)
        write_lemm.writerows(reddit_lemm)
        
    file_stem.close()
    file_lemm.close()
    df['Lemmatization'] = reddit_lemm
    df['Stemming'] = reddit_stem
    return df


    
if __name__ == "__main__":
    #Data Proprocessing
    df = pd.read_csv('./Data/500_Reddit_users_posts_labels.csv')
    df = tokenized(df)
    df = lemmAndStem(df)
