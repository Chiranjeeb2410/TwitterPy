from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r"test_pre.csv", encoding ="ISO-8859-1")
comments= ' '
stopwords = set(STOPWORDS)

#wordcloud visuals for neg_tweets
neg_tweets = df[df.pred == 0]
neg_string = []
for t in neg_tweets.sentimenttext:
    neg_string.append(t)
neg_string = pd.Series(neg_string).str.cat(sep=' ')

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(neg_string)

# plot WordCloud image
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()
