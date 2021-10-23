from numpy import mat
import pandas as pd
from csv import reader
import math

from pandas.io.parsers import TextFileReader


token_post = []
for i in range(500):
    # In each iteration, add an empty list to the main list
    token_post.append([])

# open file in read mode
i = 0
with open('reddit_lemm.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        token_post[i] = row
        i += 1

#print(token_post[-1])
#print(len(token_post))


# Creating the Bag of Words model 
#wordCount is a dictionary, where its keys are all the unique
#words in the 500 comments, its values are the number of times
#each word appears in all the comments
wordCount = {} 
for data in token_post: 
    #words = nltk.word_tokenize(data) 
    for word in data: 
        if word not in wordCount.keys(): 
            wordCount[word] = 1
        else: 
            wordCount[word] += 1


#print(wordCount)
#print(len(wordCount))



"""
Finding the tf values and idf values:
tfValue = numAppearPerComment/numInComment
idfValue = log(500/w.value)
"""
idfList = []
totalComments = len(token_post)
#print(totalComments)
for w in wordCount.values():
    idfValue = math.log(totalComments/w)
    idfList.append(idfValue)

#print(idfList)

first_matrix_token = []
for j in wordCount.keys():
    first_matrix_token.append(j)

matrix_token = []

for i in range(501):
    # In each iteration, add an empty list to the main list
    if (i == 0):
        matrix_token.append(first_matrix_token)
    else:
        matrix_token.append([0]*len(wordCount))


commend = 1
for m_data in token_post:
    for m_word in m_data:
        if m_word in matrix_token[0]:
            index = 0
            for i in matrix_token[0]:
                if (i == m_word):
                    break
                index += 1
            matrix_token[commend][index] += 1
    commend += 1

#tfValue = numAppearPerComment/numInComment
tfList = []
for i in range(len(matrix_token)):
    for j in range(len(matrix_token[i])):
        if i != 0:
            tfValue = matrix_token[i][j] / len(token_post[i-1])
            tfList.append(tfValue)
print(tfList)



tf_idf_list = []
idf_time = 0
idf_len = len(idfList)
for i in range(len(tfList)):
    tf_idf = tfList[i]*idfList[i - (i//idf_len)*idf_len]
    tf_idf_list.append(tf_idf)
print(tf_idf_list)