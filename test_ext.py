import csv
import twitter

#Variables that contains the user credentials to access Twitter API
access_token = " "
access_token_secret = " "
consumer_key = " "
consumer_secret = " "

api = twitter.Api(
    consumer_key=consumer_key,
    consumer_secret=consumer_secret,
    access_token_key=access_token,
    access_token_secret=access_token_secret)

hashtags_to_track = [
    "#metoo",
]

languages = ["en"]

stream = api.GetStreamFilter(track=hashtags_to_track, languages=languages)

with open('tweet_csv/test_pre.csv', 'w+') as csv_file:
    csv_writer = csv.writer(csv_file)
    for line in stream:
        # Signal that the line represents a tweet
        if 'in_reply_to_status_id' in line:
            tweet = twitter.Status.NewFromJsonDict(line)
            print(tweet.id)
            row = [tweet.id, tweet.user.screen_name, tweet.text]
            csv_writer.writerow(row)
            csv_file.flush()
