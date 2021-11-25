import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
import scipy.stats as stats
import pylab as pl


import re

def extract_special_feature():
    # read the data
    df0 = pd.read_csv('500_Reddit_users_posts_labels.csv')
    # delete empty lines
    df0.dropna(inplace = True)
    # declare and initialize punctuation
    search1 ="!"
    search2 = "\?"
    # count of occurrence of feature and creating new column
    df0["count!"]= df0["Post"].str.count(search1, re.I)
    df0["count?"]= df0["Post"].str.count(search2, re.I)
    df0['Uppercase'] = df0['Post'].str.findall(r'[A-Z]').str.len()
    print(df0)

    h1 = np.sort(df0["count!"])
    mean = statistics.mean(h1)
    sd = statistics.stdev(h1)
    plt.plot(h1, norm.pdf(h1, mean, sd), label='for !')


    h2 = np.sort(df0["count?"])
    mean2 = statistics.mean(h2)
    sd2 = statistics.stdev(h2)
    plt.plot(h1, norm.pdf(h2, mean2, sd2), label='for ?')

    plt.ylabel('normal distribution')
    plt.legend()
    plt.show()

    labels = ['Supportive',"Idealtion","Behavior","Indicator","Attempt"]
    Supportive = 0
    Ideation = 0
    Behavior = 0
    Indicator = 0
    Attempt = 0
    sumofexclamation = [0, 0, 0, 0, 0]
    sumofques = [0, 0, 0, 0, 0]
    row = len(df0)
    for i in range(row):
        if df0['Label'][i] == "Supportive":
            Supportive += 1
            sumofexclamation[0] += df0["count!"][i]
            sumofques[0] += df0["count?"][i]
        elif df0['Label'][i] == "Ideation":
            Ideation += 1
            sumofexclamation[1] += df0["count!"][i]
            sumofques[1] += df0["count?"][i]
        elif df0['Label'][i] == "Behavior":
            Behavior += 1
            sumofexclamation[2] += df0["count!"][i]
            sumofques[2] += df0["count?"][i]
        elif df0['Label'][i] == "Indicator":
            Indicator += 1
            sumofexclamation[3] += df0["count!"][i]
            sumofques[3] += df0["count?"][i]
        elif df0['Label'][i] == "Attempt":
            Attempt += 1
            sumofexclamation[4] += df0["count!"][i]
            sumofques[4] += df0["count?"][i]
    
    Labels = ['Supportive','Ideation','Behavior','Indicator','Attempt']
    Exclamation_mark_Per_labels = [sumofexclamation[0]/Supportive,sumofexclamation[1]/Ideation,sumofexclamation[2]/Behavior,sumofexclamation[3]/Indicator,sumofexclamation[4]/Attempt]
    Question_mark_Per_labels = [sumofques[0]/Supportive,sumofques[1]/Ideation,sumofques[2]/Behavior,sumofques[3]/Indicator,sumofques[4]/Attempt]

    X_axis = np.arange(len(Labels))

    plt.bar(X_axis - 0.2, Exclamation_mark_Per_labels, 0.4, label = 'Exclamation Mark (!)')
    plt.bar(X_axis + 0.2, Question_mark_Per_labels, 0.4, label = 'Question Mark (?)')

    plt.xticks(X_axis, Labels)
    plt.title('Labels Vs Average Number of Marks in each Label')
    plt.xlabel('Labels')
    plt.ylabel('Average Number of Marks in each Label')
    plt.legend()
    plt.show()
    


    
    return df0

extract_special_feature()

