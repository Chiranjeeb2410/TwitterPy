Project template for Twitter Sentiment Analysis using Machine Learning and Python.

This project particularly aims to analyze and classify the sentiments associated with the tweets extracted through the application of the Twitter Streaming API using Python and its related modules. The implementation overview is as
follows:

1. Downloading raw pre-defined Twitter datasets without using the API for preprocessing.

2. Preprocess the targeted texts for each tweet and clean up the data by removing short words,
   punctuations, special characters, user handles, hashtags etc.

3. Design a tf-idf vectorizer to transform the preprocessed text to feature vectors to be used
   as input to estimator/classifier by using the previous dataset as training data.
4. Feed the created feature vectors and associated labels to a linear SVM classifier.
5. Extract a signficant Twitter dataset using the Streaming API to be used as the testing data
   and thereby predict the sentiments(positive: 1/negative: 0) corresponding to each tweet text
   and outputting the feature matrix for the same.
6. Displaying the predicted output in the following ways:

      1. Percentage count and pie-chart representation of positive/negative
         tweets for testing data.
      2. Total number of positive/negative tweets for the same.
      3. Wordcloud representations of keywords corresponding to positive/
         negative tweet datasets.
