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

    
    return df0

extract_special_feature()

