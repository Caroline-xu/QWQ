
####################### 
# Wanqing Part:Stemming And Lemmatization #######################
# this file contains stemming and lemmatization to the posts
# after cleaning punctuations and transfer all digits to lowercase
# it would push stemming & lemming posts into 2 different csv files() 
import csv
import pandas as pd
import nltk
#Lemmatization
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import WordPunctTokenizer

#######need to change
#open post file after cleaning(should be replaced by no punctuation and no uppercase version)
posts = open('posts.csv','r')

##lemmatization
lemmatizer = WordNetLemmatizer()
##stemming
stemmer = PorterStemmer()
#create a lemmatization list for reddit posts
reddit_lemm = []
#create a stemming list for reddit posts
reddit_stem = []
#this is avoid to save title "posts" into our new lists!
topic = posts.readline()
######need to change this hardcode(500) to the length of post !!!!!!
for i in range(500):
    #initilize single post list
    singlePost_stem = []
    singlePost_lemm = []
    #tokenize the csv file of sentences, and read file line by line
    tokenization = nltk.word_tokenize(posts.readline())
    for w in tokenization:
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

#create csv file for stemming posts
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
 
#if you want to see display of after lemmatization you could cancel comment on this:    
#display lemmatization
'''print("lemmatization:") 
count = 1
for i in reddit_lemm:
    print(count,i)
    count += 1'''

##display stemming
'''print("stemming:")  
count = 1
for i in reddit_stem:
    print(count,i)
    count += 1'''
    


