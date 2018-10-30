import numpy as np
import pandas as pd
import re
import warnings

#visualization
import matplotlib.pyplot as plt
import matplotlib
#import seaborn as sns
#from Ipython.display import display
#from mpl_toolkits.basemap import Basemap
#from wordcloud import WordCloud, STOPWORDS

#nltk
#from nltk.stem import WordNetLemmatizer
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from nltk.sentiment.util import *
#from nltk import tokenize

matplotlib.style.use('ggplot')
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

tweets = pd.read_csv('tweets_all.csv', encoding = "ISO-8859-1")

tweets['handles'] =  ''

for i in range(len(tweets['text'])):
    try:
        tweets['handles'][i] = tweets['text'].str.split(' ')[i][0]
    except AttributeError:
        tweets['handles'][i] = 'other'

for i in range(len(tweets['text'])):
    if tweets['handles'].str.contains('@')[i]  == False:
        tweets['handles'][i] = 'other'

for i in range(len(tweets['text'])):
    tweets['text'][i] = " ".join([word for word in tweets['text'][i].split()
                                if 'http' not in word and '@' not in word and '<' not in word])

tweets['text'] = tweets['text'].apply(lambda x: re.sub('[!@#$:).;,?&]', '', x.lower()))
tweets['text'] = tweets['text'].apply(lambda x: re.sub('  ', ' ', x))

test = tweets['text'][1]
print (test)


