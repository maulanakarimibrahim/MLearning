# Preparing Class
import random


class Sentiment:
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"


class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else:  # Score of 4 or 5
            return Sentiment.POSITIVE

class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews

    def get_text(self):
        return [x.text for x in self.reviews]

    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]

    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))
        positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))
        positive_shrunk = positive[:len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)

# Load the Data
import json

file_name = 'C:/Users/workm/learningsource/books_small_10000.json'

reviews = []
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall']))

reviews[5].text

print(reviews[5].text)
print('')

# Preparing the Data
from sklearn.model_selection import train_test_split

training, test = train_test_split(reviews, test_size=0.33, random_state=40)

train_container = ReviewContainer(training)
test_container = ReviewContainer(test)

train_container.evenly_distribute()
train_x = train_container.get_text()
train_y = train_container.get_sentiment()

test_container.evenly_distribute()
test_x = test_container.get_text()
test_y = test_container.get_sentiment()

print(train_y.count(Sentiment.POSITIVE))
print(train_y.count(Sentiment.NEGATIVE))


# Transform words into Matrices through Bag of Words Vectorization
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit_transform(train_x)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)

test_x_vectors = vectorizer.transform(test_x)

print(train_x[1])
print(train_x_vectors[1].toarray())
print('')

# Choosing Classifier to classify data
# In this case we choose Linear SVM

from sklearn import svm
clf_svm = svm.SVC(kernel='linear')

clf_svm.fit(train_x_vectors, train_y)

print(test_x[0])

clf_result = clf_svm.predict(test_x_vectors[0])
print(clf_result)
print('')

# Result Evaluation

clf_evresult = clf_svm.score(test_x_vectors, test_y)
print(clf_evresult)
print('')

from sklearn.metrics import f1_score

f1s = f1_score(test_y, clf_svm.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE])
print(f1s)
print('')

print(train_y[0:10])
print(train_y.count(Sentiment.POSITIVE))
print(train_y.count(Sentiment.NEUTRAL))
print(train_y.count(Sentiment.NEGATIVE))

test_set = ['I thoroughly enjoyed this, 5 stars', 'Bad book do not buy', 'horrible waste of time', 'worst bitch']
new_test = vectorizer.transform(test_set)

print(clf_svm.predict(new_test))