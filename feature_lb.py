import pandas as pd
import csv
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.util import ngrams
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

def feature_lb():
    from preprocessing import reddit_lemm
    from preprocessing import reddit_stem
    from preprocessing import stop_words
    from preprocessing import df

    from sklearn.feature_extraction.text import TfidfVectorizer

    
    #Label = df['New Label']
    
    #TF-IDF
    tfidf = TfidfVectorizer(analyzer=lambda x:[w for w in x if w not in stop_words])
    tfidf_vectors = tfidf.fit_transform(reddit_lemm)

    df2 = pd.DataFrame(tfidf_vectors.todense(), columns=tfidf.vocabulary_)
    
    #compression_opts = dict(method='zip',
                            #archive_name='tfidf.csv') 
    #df2.to_csv('tfidf.zip', index=False,
            #compression=compression_opts) 

    #CountVectorizer word Frequency
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(analyzer=lambda x:[w for w in x if w not in stop_words])
    cv_vectors = cv.fit_transform(reddit_lemm)


    df3 = pd.DataFrame(cv_vectors.todense(), columns=cv.vocabulary_)
    compression_opts = dict(method='zip',
                            archive_name='countvec.csv') 
    df3.to_csv('countvec.zip', index=False,
            compression=compression_opts) 

    ##ngram
    vecn = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    for n in vecn:
        n_grams = []
        tfidf_ngrams = TfidfVectorizer(analyzer=lambda x:[w for w in x if w not in stop_words])
        for i in range(500):
            n_grams.append(ngrams(reddit_lemm[i], n))

        tfidf_ngrams_vectors = tfidf_ngrams.fit_transform(n_grams)
        dfngram = pd.DataFrame(tfidf_ngrams_vectors.todense(), columns=tfidf_ngrams.vocabulary_)



    #Sentiment extraction
    #polarity of sentences
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
        
    def getPolarity():
        df = pd.read_csv('500_Reddit_users_posts_labels.csv')
        polarity_score = []
        #df["Post"] = df.apply(axis=1)
        for i in range(500):
            this_score = sentiment_scores(df["Post"][i])
            polarity_score.append(this_score)
            #print("index", i, "sentiment polarity:", this_score,"\n")
        return polarity_score
        
    polarity_score = getPolarity()
    #print(polarity_score)
    #add polarity to dataframe
    df2['Polarity'] = polarity_score
    #add label to data frame
    df2['New Label'] = df['Label'].replace(['Supportive','Ideation','Attempt','Behavior','Indicator'],[0, 1, 2, 3, 4])
    print(df2)

    '''#n-gram
    all_ngrams = []
    for sent in reddit_lemm:
        all_ngrams.extend(nltk.ngrams(sent, len(sent)))'''

        
    return df2
