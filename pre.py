import numpy as np
import pandas as pd
import re, string
import warnings
import sklearn
import fileinput

#visualization
import matplotlib.pyplot as plt
import matplotlib

from sklearn import linear_model
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords
from glob import glob
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from collections import Counter
from sklearn.metrics import accuracy_score
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVC
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import confusion_matrix

matplotlib.style.use('ggplot')
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

tweets = pd.read_csv('tweet_csv/new1.csv', encoding = "ISO-8859-1")
tweets['handles'] =  ''

#remove handles
for i in range(len(tweets['SentimentText'])):
    try:
        tweets['handles'][i] = tweets['SentimentText'].str.split(' ')[i][0]
    except AttributeError:
        tweets['handles'][i] = 'other'
#len(tweets['text'])

#Preprocessing handles. select handles contains 'RT @'
for i in range(len(tweets['SentimentText'])):
    if tweets['handles'].str.contains('@')[i]  == False:
        tweets['handles'][i] = 'other'

# remove URLs, RTs, and twitter handles
for i in range(len(tweets['SentimentText'])):
    tweets['SentimentText'][i] = " ".join([word for word in tweets['SentimentText'][i].split()
    if 'http' not in word and '@' not in word and '<' not in word])

#remove special chars/numbers/hashtags/short words/stopwords and adds tokenization
tweets['SentimentText'] = tweets['SentimentText'].apply(lambda x: re.sub('[!@$:).;,?&]', '', x.lower()))
tweets['SentimentText'] = tweets['SentimentText'].str.replace("[^a-zA-Z#]", " ")
tweets['SentimentText'] = tweets['SentimentText'].apply(lambda x: re.sub(r'\B(\#[a-zA-Z]+\b)', '', x.lower()))
tweets['SentimentText'] = tweets['SentimentText'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
tokens = tweets['SentimentText'].apply(lambda x: x.split())
stop = stopwords.words('english')
tweets['SentimentText'] = tweets['SentimentText'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#stemming
ps = PorterStemmer()
tweets['SentimentText'] = tweets['SentimentText'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))

#outputting text segments from tweets in pre_tweets.csv
with open('tweet_csv/pre_tweets.csv', "w") as outfile:
	for entries in tweets['SentimentText']:
		outfile.write(entries)
		outfile.write("\n")

#adds sentiment values corresponding to each tweet text
with open('tweet_csv/pre_tweets.csv', "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerows(zip(tweets['Sentiment'], tweets['SentimentText']))

#adds Sentiment and text header labels
for line in fileinput.input(files=['tweet_csv/pre_tweets.csv'], inplace=True):
    if fileinput.isfirstline():
        print ('Sentiment, Text')
    print (line),

#building tf-idf vectorizer/linearSVC
x_train, x_test, y_train, y_test = train_test_split(tweets["SentimentText"],
    tweets["Sentiment"], test_size = 0.3, random_state = 2)
count_vect = CountVectorizer(stop_words='english')
transformer = TfidfTransformer(norm='l2',sublinear_tf=True)

X_train_counts = count_vect.fit_transform(x_train)
X_train_tfidf = transformer.fit_transform(X_train_counts)
x_test_counts = count_vect.transform(x_test)
x_test_tfidf = transformer.transform(x_test_counts)

#using/training linear SVM model
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

#testing model and accuracy
print("\n")
predictions = model.predict(x_test_tfidf)
print(predictions)
print(accuracy_score(y_test, predictions))

#actual testing with data extracted through Twitter api and preprocesed
test_data = pd.read_csv("tweet_csv/test_pre.csv")

## for transforming the test data ##
test_counts = count_vect.transform(test_data['sentimenttext'])
test_tfidf = transformer.transform(test_counts)

## predicting the results ##
print("\n")
predictions1 = model.predict(test_tfidf)
print(predictions1)
print("\n")

#adding predicted sentiments within the header
df = pd.read_csv("tweet_csv/test_pre.csv")
df ['pred'] = predictions1
df.to_csv('tweet_csv/test_pre.csv')

#percentage analysis of extracted tweets
pos = 0
neg = 0

for i in predictions1:
  if i == 1:
    pos= pos+1
  elif i == 0:
    neg = neg+1

pos_count = pos*100/(pos+neg)
neg_count = neg*100/(pos+neg)

print("Percentage positive tweets: ", pos_count)
print("Percentage negative tweets: ", neg_count)
print("\n")

#count of positive/negative tweets from test data
with open('tweet_csv/test_pre.csv', 'r') as count_res:
  counter = Counter()
  for row in csv.DictReader(count_res):
    counter[row['pred']] += 1

  positive = counter['1']
  negative = counter['0']

print("No. of positive tweets: ", positive)
print("No. of negative tweets: ", negative)
print("\n")

colors = ['slateblue', 'darkblue']
sizes = [positive, negative]
labels = 'Positive: 32.03%', 'Negative: 67.97%'

## use matplotlib to plot the chart
plt.pie(
   x=sizes,
   shadow=True,
   colors=colors,
   labels=labels,
   startangle=90
)

plt.title("Sentiment of {} Tweets about {}".format(200, '#MeToo'))
plt.show()


