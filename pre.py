import numpy as np
import pandas as pd
import re
import warnings
import json
from pprint import pprint

#visualization
import matplotlib.pyplot as plt
import matplotlib
#import seaborn as sns
#from Ipython.display import display
#from mpl_toolkits.basemap import Basemap

#nltk
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize
from nltk.stem.porter import *

matplotlib.style.use('ggplot')
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

tweets = pd.read_csv('tweets_all.csv', encoding = "ISO-8859-1")
tweets['handles'] =  ''

for i in range(2): #range(len(tweets['text']))
    try:
        tweets['handles'][i] = tweets['text'].str.split(' ')[i][0]
    except AttributeError:
        tweets['handles'][i] = 'other'

for i in range(2):
    if tweets['handles'].str.contains('@')[i]  == False:
        tweets['handles'][i] = 'other'

for i in range(2):
    tweets['text'][i] = " ".join([word for word in tweets['text'][i].split()
                                if 'http' not in word and '@' not in word and '<' not in word])

tweets['text'] = tweets['text'].apply(lambda x: re.sub('[!@#$:).;,?&]', '', x.lower()))
tweets['text'] = tweets['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
tweets['text'] = tweets['text'].str.replace("[^a-zA-Z#]", " ")
tokens = tweets['text'].apply(lambda x: x.split())

#stemming
#stemmer = PorterStemmer()

#tokens = tokens.apply(lambda x: [stemmer.stem(i) for i in x])
#tokens.head()

#for i in range(2):
    #tokens[i] = ' '.join(tokens[i])

#tweets['text'] = tokens

test = tweets['text']
print (test)

with open('pre_tweets.csv','w') as f:
    for line in test:
        f.write(line)
        f.write('\n')

