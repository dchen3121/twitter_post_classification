import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 11099 total tweets
all_tweets = pd.read_json("random_tweets.json", lines=True)

# setting bar for "viral" tweets
top_ten_percent_retweets = all_tweets['retweet_count'].quantile(0.9)
all_tweets['is_viral'] = np.where(
    all_tweets['retweet_count'] >= top_ten_percent_retweets, 1, 0)

# making features
all_tweets['tweet_length'] = all_tweets.apply(
    lambda tweet: len(tweet['text']), axis=1)
all_tweets['followers_count'] = all_tweets.apply(
    lambda tweet: tweet['user']['followers_count'], axis=1)
all_tweets['friends_count'] = all_tweets.apply(
    lambda tweet: tweet['user']['friends_count'], axis=1)

# normalizing data
data_labels = all_tweets['is_viral']
data = all_tweets[['tweet_length', 'followers_count', 'friends_count']]
scaled_data = scale(data, axis=0)

training_data, testing_data, training_labels, testing_labels = train_test_split(
    scaled_data, data_labels, test_size=0.2, random_state=1)

max_score, max_k = 0, 0
for k in range(1, 200):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    score = classifier.score(testing_data, testing_labels)
    if score > max_score:
        max_score, max_k = score, k

classifier = KNeighborsClassifier(n_neighbors=max_k)
classifier.fit(train_data, train_labels)
