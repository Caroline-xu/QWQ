import pandas as pd

#There are 3 steps while creating a BoW model :
#1. The first step is text-preprocessing which involves:
####1. converting the entire text into lower case characters. -> done
####2. removing all punctuations and unnecessary symbols. -> done
#2. The second step is to create a vocabulary of all unique words from the corpus

token_post = pd.read_csv('reddit_lemm.csv')
l = len(token_post)

vocabulary_list = []
for i in range(l):
    # In each iteration, add an empty list to the main list
    vocabulary_list.append([])

##find the index of the word in vocabulary_list
find_index(word, vocabulary_list)


for i in range(l):
    for word in token_post[i]:
        if word in vocabulary_list[0]:
            