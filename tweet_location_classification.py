import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

new_york_tweets, london_tweets, paris_tweets = (
    pd.read_json("new_york.json", lines=True),
    pd.read_json("london.json", lines=True),
    pd.read_json("paris.json", lines=True)
)

# Naive Bayes Classifier

new_york_text, london_text, paris_text = (
    new_york_tweets["text"].tolist(),
    london_tweets["text"].tolist(),
    paris_tweets["text"].tolist()
)

all_tweets = new_york_text + london_text + paris_text
all_tweets_labels = [0] * len(new_york_text) + \
    [1] * len(london_text) + [2] * len(paris_text)

# 10059 training data, 2515 testing data
training_data, testing_data, training_labels, testing_labels = train_test_split(
    all_tweets, all_tweets_labels, test_size=0.2, random_state=1)

counter = CountVectorizer()
counter.fit(training_data)
training_counts = counter.transform(training_data)
testing_counts = counter.transform(testing_data)

classifier = MultinomialNB()
classifier.fit(training_counts, training_labels)

# predictions = classifier.predict(testing_counts)
# print(accuracy_score(testing_labels, predictions))
# print(confusion_matrix(testing_labels, predictions))

print("Enter the tweet which you'd like to know the location of:")
test_tweet = input()
tweet_counts = counter.transform([test_tweet])
print(tweet_counts)
result_location = {0: 'New York', 1: 'London', 2: 'Paris'}[
    classifier.predict(tweet_counts)[0]]
print(f"The classifier guesses that the tweet location is {result_location}.")
