# this file includes
#1. bar chart of label 
#2. polarity chart of label
import pandas as pd
import matplotlib.pyplot as plot
from feature_lb import feature_lb
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def getLabelNum():
    df = pd.read_csv('500_Reddit_users_posts_labels.csv')
    index = df.index
    row = len(index)
    Supportive = 0
    Ideation = 0
    Behavior = 0
    Indicator = 0
    Attempt = 0
    for i in range(row):
        if df['Label'][i] == "Supportive":
            Supportive += 1
        elif df['Label'][i] == "Ideation":
            Ideation += 1
        elif df['Label'][i] == "Behavior":
            Behavior += 1
        elif df['Label'][i] == "Indicator":
            Indicator += 1
        elif df['Label'][i] == "Attempt":
            Attempt += 1
    return Supportive,Ideation,Behavior,Indicator,Attempt

def plotBarChart():
    Supportive,Ideation,Behavior,Indicator,Attempt = getLabelNum()
    data = {"Label":['Supportive',"Idealtion","Behavior","Indicator","Attempt"],
            "Number Of Posts":[Supportive,Ideation,Behavior,Indicator,Attempt]}
    dfBarChart = pd.DataFrame(data=data);
    dfBarChart.plot.bar(x="Label", y="Number Of Posts", rot=70, title="Label Classification");
    plot.show(block=True);  

def getLabelPolarity():
    from feature_lb import feature_lb
    df = feature_lb()
    index = df.index
    row = len(index)
   #print(df)
    neg = [0,0,0,0,0]
    pos = [0,0,0,0,0]
    for i in range(row):
        #print("i:",i)
        #print("New label is: ", df['New Label'][i])
        if df['New Label'][i] == 0:
            #print("in label 0:")
            if df['Polarity'][i] == 1:
                #print("this label is pos")
                pos[0] += 1
            elif df['Polarity'][i] == -1: 
                neg[0] += 1
        elif df['New Label'][i] == 1:
            if df['Polarity'][i] == 1:
                pos[1] += 1
            elif df['Polarity'][i] == -1: 
                neg[1] += 1
        elif df['New Label'][i] == 2:
            if df['Polarity'][i] == 1:
                pos[2] += 1
            elif df['Polarity'][i] == -1: 
                neg[2] += 1
        elif df['New Label'][i] == 3:
            if df['Polarity'][i] == 1:
                pos[3] += 1
            elif df['Polarity'][i] == -1: 
                neg[3] += 1
        elif df['New Label'][i] == 4:
            if df['Polarity'][i] == 1:
                pos[4] += 1
            elif df['Polarity'][i] == -1: 
                neg[4] += 1
        
    return neg,pos

def plotPolarity():  
    neg,pos = getLabelPolarity()
    real_neg = [ -x for x in neg]
    labels = ['Supportive',"Idealtion","Behavior","Indicator","Attempt"]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=neg,
                    base= real_neg,
                    marker_color='crimson',
                    name='negative'))
    fig.add_trace(go.Bar(x=labels, y=pos,
                    base=0,
                    marker_color='lightslategrey',
                    name='positive'
                    ))

    fig.show()
    
if __name__ == '__main__':        
    plotBarChart()
    plotPolarity()
             
                
    
