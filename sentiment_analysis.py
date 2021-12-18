import nltk
import numpy as np
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import json
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
nltk.download('vader_lexicon')
import re
from tqdm import tqdm
tqdm.pandas()
import plotly.graph_objects as go
pd.options.plotting.backend = "plotly"

def tokenize_message(message) :
    tokens = word_tokenize(message)
    tokens =  [token.lower() for token in tokens]
    return tokens

def remove_stop_keywords(message) :
    stopwords_set = set(stopwords.words('english'))
    clean_message = [word for word in message if word not in stopwords_set]
    return clean_message

def get_text(text) :
    final = ""
    if isinstance(text, str) :
        final = text
    if isinstance(text, list) :
        for msg in text:
            if isinstance(msg, str) :
                final = final + " " + msg
                
    final = final.lower()
    if 'shiba' not in final and 'doge' not in final :
        return ""
    if final.isascii() :
        return final
    else :
        return ""
    return final

def get_polarity_scores(message) :
    tokens = tokenize_message(message)
    non_stop_keywords = remove_stop_keywords(tokens)
    sia = SentimentIntensityAnalyzer()
    scores =  sia.polarity_scores(" ".join(non_stop_keywords))
    return scores['compound']

def find_sentiment(score) :
    if score >= 0.05 :
        return "Positive"
    elif score <= -0.05 :
        return "Negative"
    else :
        return "Neutral"

with open("result.json") as file:
    data = json.load(file)

messages = data['messages']
df = pd.json_normalize(messages)
tqdm.pandas(desc='Predicting sentiment of messages')

df['text'] = df['text'].apply(get_text)

filtered_rows = df[df['text'] != ""]
filtered_rows['compound_score'] = filtered_rows['text'].progress_apply(lambda x : get_polarity_scores(x))
filtered_rows['sentiment'] = filtered_rows['compound_score'].apply(lambda x : find_sentiment(x))
df1 = filtered_rows[['sentiment', 'text', 'date', 'compound_score']]

df1.loc[:,'date'] = df1['date'].apply(pd.to_datetime)
df2 = df1.groupby(df1['date'].dt.date)['sentiment'].value_counts()
df3 = df1.groupby(df1['date'].dt.date)['compound_score'].mean()

date = []
positive = []
negative = []
neutral = []
message_count = {}
for i in range(len(df2.index)):
    row = df2.index[i]
    sentiments = ["Positive", "Negative", "Neutral"]
    date.append(row[0])
    if row[1] == "Positive" :
        positive.append(df2[i])
        sentiments.remove("Positive")
    elif row[1] == "Neutral" :
        neutral.append(df2[i])
        sentiments.remove("Neutral")
    else :
        negative.append(df2[i])
        sentiments.remove("Negative")
    if row[0] in message_count :
        message_count[row[0]] = message_count[row[0]] + df2[i]
    else :
        message_count[row[0]] = df2[i]
    for sentiment in sentiments :
        if sentiment == "Positive":
            positive.append(0)
        elif sentiment == "Negative" :
            negative.append(0)
        else :
            neutral.append(0)
X_axis = np.arange(len(date))
fig = go.Figure(data=[
    go.Bar(name='Positive', x=date, y=positive, marker_color = 'green'),
    go.Bar(name='Neutral', x=date, y=neutral, marker_color = 'blue'),
    go.Bar(name='Negative', x=date, y=negative, marker_color = 'red')
])
fig.update_layout(barmode='group')
fig.update_layout(xaxis = dict(tickmode = 'linear'), title = "No. of messages of each sentiment per day")
fig.show()
fig1 = go.Figure(data = [go.Bar(name = " No. of messages", x = list(message_count.keys()), y = list(message_count.values()))])
fig1.update_layout(xaxis = dict(tickmode = 'linear'), title = "No. of messages per day")
fig1.show()
fig2 = df3.plot()
fig2.update_layout(xaxis = dict(tickmode = 'linear'), title = "Average sentiment score per day")
fig2.show()