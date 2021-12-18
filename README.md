**Instructions to run the code**

To run the code use this command "python3 sentiment_analysis.py"
The telegram messages data is read from "result.json" file present in the repo. result.json and sentiment_analysis.py should be in same directory before running the code.

**Overview**

From the mentioned telegram group I have exported the data in to json file.
I have parsed the result.json file to capture messages in to a pandas data frame. In the json file the messages are present in the 'text' key. If messages have any links I have ignored the links and merged the remaining parts of the message
I have used SentimentIntensityAnalyzer from nltk package to predict the sentiment of the message.
Before sending the message to the analyzer I have done few preprocessing steps
I have first filtered out messages containing the mentioned keywords 'shiba' and 'doge'. Of these messages I have removed non english messages. This is done get_text() method.
After filtering out messages with above conditions I have toeknized the message using nltk tokenizer to reomve stop words from the text. This is done in tokenize_message() and remove_stop_words() method.
To remove stop words I have used stopwords from the corpus available in nltk. (nltk.stopwords)
After doing above prerpocessing the message is then sent to SentimentIntensityAnalyzer of nltk to get the polarity_scores of the message.
I have considered cpmpound_score from the polarity_scores to determine the sentiment of the message. I have stored sentiment of each message in "sentiment" column of the data frame. It has either positive, neutral or negative values. The ranges are choosen as folllows : if compound_score >=0.05 it is positive, if compound_score <= 0 it is negative and else it is neutral.
After computing the sentiment score of each message I have calculated average sentiment of each day by computing average of compound scores for each day. To achieve this I have aggregated pandas data fram on date object and calculated mean from pandas inbuilt method mean() on data frame.
I have plotted the averaged copound score over each day using bar graphs in plotly. This is plot_1.png in the repo
Similarly I have aggregated no. of messages per day and plotted using plotly. This is plot_2.png in the repo
I have also generated plot for no. of messages of each sentiment per day. This is plot_3.png in the repo.

**Summary of results**

There are total 2403 english messages from May 1, 2021 to May 15, 2021 which contains wither of these keywords "shib" or "doge"
The sentiment initially starts from positive on May 1 and fluctuates in the positive region up to May 4 after which there is a dip in its value. It starts to decline from Positive to Neutral to Negative  from May 4th to May 5th to 6th.
The sentiment then starts to raise to positive from May 7th and then fluctuates for a few days between Positive and Neutral between May 8th and May12th
It then continues to rise from 13th till 15th of May just about reaching its initial value and reching Positive sentiment
Highest number of messages were recorded related to these keywords (around 600 messages per day) on May 8th and May 10th.
