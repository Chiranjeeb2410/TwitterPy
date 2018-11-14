import numpy as np
import pandas as pd
import re, string
import warnings
import sklearn
import fileinput

#visualization
import matplotlib.pyplot as plt
import matplotlib
#import seaborn as sns
#from Ipython.display import display
#from mpl_toolkits.basemap import Basemap

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
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#from sklearn.svm import SVC
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

matplotlib.style.use('ggplot')
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

#adds Sentiment and text header labels
for line in fileinput.input(files=['test_pre.csv'], inplace=True):
    if fileinput.isfirstline():
        print ('SentimentText')
    print (line),

tweets = pd.read_csv('test_pre.csv', encoding = "ISO-8859-1")
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
tweets['SentimentText'] = tweets['SentimentText'].apply(lambda x: re.sub('[!@$:).;,?&#]', '', x.lower()))
tweets['SentimentText'] = tweets['SentimentText'].str.replace("[^a-zA-Z]", " ")
tweets['SentimentText'] = tweets['SentimentText'].apply(lambda x: re.sub(r'\B(\#[a-zA-Z]+\b)', '', x.lower()))
tweets['SentimentText'] = tweets['SentimentText'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
tokens = tweets['SentimentText'].apply(lambda x: x.split())
stop = stopwords.words('english')
tweets['SentimentText'] = tweets['SentimentText'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#stemming
ps = PorterStemmer()
tweets['SentimentText'] = tweets['SentimentText'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))

#outputting text segments from tweets in pre_tweets.csv
with open('test_pre.csv', "w") as outfile:
    for entries in tweets['SentimentText']:
        outfile.write(entries)
        outfile.write("\n")

