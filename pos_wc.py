from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r"tweet_csv/test_pre.csv", encoding ="ISO-8859-1")
comments= ' '
stopwords = set(STOPWORDS)

#wordcloud visuals for pos_tweets
pos_tweets = df[df.pred == 1]
pos_string = []
for t in pos_tweets.sentimenttext:
    pos_string.append(t)
pos_string = pd.Series(pos_string).str.cat(sep=' ')

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(pos_string)

# plot WordCloud image
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()
