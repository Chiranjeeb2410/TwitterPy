import csv

#Variables that contains the user credentials to access Twitter API
access_token = "Enter access_token"
access_token_secret = "Enter access secret"
consumer_key = "Enter consumer key"
consumer_secret = "Enter consumer secret"

import twitter
api = twitter.Api(
    consumer_key=consumer_key,
    consumer_secret=consumer_secret,
    access_token_key=access_token,
    access_token_secret=access_token_secret)

hashtags_to_track = [
    "#mood",
]

stream = api.GetStreamFilter(track=hashtags_to_track)

with open('test_tweets.csv', 'w+') as csv_file:
    csv_writer = csv.writer(csv_file)
    for line in stream:
        # Signal that the line represents a tweet
        if 'in_reply_to_status_id' in line:
            tweet = twitter.Status.NewFromJsonDict(line)
            print(tweet.id)
            row = [tweet.id, tweet.user.screen_name, tweet.text]
            csv_writer.writerow(row)
            csv_file.flush()
