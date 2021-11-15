
from preprocessing import tokenized
import pandas as pd

def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(zip(wordlist,wordfreq))

def sortFreqDict(freqdict):
    freqDic = [(freqdict[key], key) for key in freqdict]
    freqDic.sort()
    freqDic.reverse()
    return freqDic



def word_freq(label):
    wordsList = []
    for value in label:
        wordsList = wordsList + value
    freqList = sortFreqDict(wordListToFreqDict(wordsList))[0:15]
    print("Total word count length:", len(wordsList))
    print("Top 15 most common words in Label:")
    for i in freqList:
        print (i)

if __name__ == '__main__':

    df = pd.read_csv('/Users/wangjiale/Desktop/QWQ/Data/500_Reddit_users_posts_labels.csv')

    df = tokenized(df)

    label_0 = df.loc[df['New Label']==0,:]['Tokenized Post']
    label_1 = df.loc[df['New Label']==1,:]['Tokenized Post']
    label_2 = df.loc[df['New Label']==2,:]['Tokenized Post']
    label_3 = df.loc[df['New Label']==3,:]['Tokenized Post']
    label_4 = df.loc[df['New Label']==4,:]['Tokenized Post']

    word_freq(label_0)