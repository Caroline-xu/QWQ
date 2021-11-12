import pandas as pd
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
    return df0



