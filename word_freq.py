
from preprocessing import tokenized
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
#nltk.download('vader_lexicon')
import pandas as pd

def wordListToFreqDict(wordlist):
    somewords = []
    for p in wordlist:
        if sentiment_scores(p) != 0:
            somewords.append(p)
    wordfreq = [somewords.count(i) for i in somewords]
    return dict(zip(somewords,wordfreq))

def sortFreqDict(freqdict):
    freqDic = [(freqdict[key], key) for key in freqdict]
    freqDic.sort()
    freqDic.reverse()
    return freqDic

def sentiment_scores(sentence): 
    #initialize polarity of now sentence
    polarity = 0
    # Create a SentimentIntensityAnalyzer object. 
    sid_obj = SentimentIntensityAnalyzer()

    # polarity_scores method of SentimentIntensityAnalyzer 
    # oject gives a sentiment dictionary. 
    # which contains pos, neg, neu, and compound scores. 
    sentiment_dict = sid_obj.polarity_scores(sentence) 
    
    #get rate of polarity of one post
    this_negative = sentiment_dict['neg']
    this_neural = sentiment_dict['neu']
    this_positive = sentiment_dict['pos']
    
    # decide sentiment as positive, negative and neutral 
    if sentiment_dict['compound'] >= 0.05 : 
        #positive
        polarity = 1 
    elif sentiment_dict['compound'] <= - 0.05 : 
        #negative
        polarity = -1
    else: 
        #neural
        polarity = 0
    return polarity


def word_freq(label):
    wordsList = []
    for value in label:
        wordsList = wordsList + value
    freqList = sortFreqDict(wordListToFreqDict(wordsList))[0:50]
    print("Total word count length:", len(wordsList))
    print("Top 15 most common words in Label:")
    for i in freqList:
        print (i)

if __name__ == '__main__':

    df = pd.read_csv('Data/500_Reddit_users_posts_labels.csv')

    df = tokenized(df)

    label_0 = df.loc[df['New Label']==0,:]['Tokenized Post']
    label_1 = df.loc[df['New Label']==1,:]['Tokenized Post']
    label_2 = df.loc[df['New Label']==2,:]['Tokenized Post']
    label_3 = df.loc[df['New Label']==3,:]['Tokenized Post']
    label_4 = df.loc[df['New Label']==4,:]['Tokenized Post']
    label_list = [label_0, label_1, label_2, label_3, label_4]


    #print(word_freq(label_0))

    for i in label_list:
        print(word_freq(i))

'''
    for i in label_0:
        for j in i: 
            print(j)
            print(sentiment_scores(j))
'''
  
    
    
