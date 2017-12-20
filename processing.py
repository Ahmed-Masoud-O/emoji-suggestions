from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
import fasttext
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import timeit


def post_append(temp_vector, number=0.0):
    extra_vector = [number] * delta
    # for i in range(delta):
    #     temp_vector.append(number)
    return temp_vector + extra_vector


def pre_append(temp_vector, number=0.0):
    zeros_vector = [number] * delta
    temp_vector = zeros_vector + temp_vector
    return temp_vector

names = ["Nearest Neighbors",
         "Linear SVM",
         "RBF SVM",
         "Decision Tree",
         "Random Forest",
         "Neural Net",
         "AdaBoost",
         "Naive Bayes",
         "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]
tokenizer = TweetTokenizer()
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')
word_vectors = []
length_vector = []
index = 0
my_labels = open('labels.txt', encoding="utf-8")
train_tweets = []
test_tweets = []
train_labels = []
test_labels = []
model = fasttext.load_model('model.bin')
start = timeit.default_timer()
limit = 1000
train_count = 700
for tweets, labels in zip(open('Output.txt', encoding="utf-8"), my_labels):
    if index == limit:
        break
    if index < train_count:
        train_labels.append(labels.strip())
    else:
        test_labels.append(labels.strip())
    index += 1
    cleaned_tweet = tweets
    tokenized_tweet = tokenizer.tokenize(cleaned_tweet)
    filtered_tweet = tokenized_tweet
    tweet_vector = []
    for idx, word in enumerate(filtered_tweet):
        filtered_tweet[idx] = lemmatizer.lemmatize(word)
        filtered_tweet[idx] = stemmer.stem(word)
        tweet_vector = tweet_vector + model[word]
    length_vector.append(len(tweet_vector))
    word_vectors.append(tweet_vector)
max_vec = max(length_vector)
index = 0
stop = timeit.default_timer()
print("tokenizing and vectorization")
print("--------------------------------------")
print(stop - start)
print("--------------------------------------")
failed = 0
start = timeit.default_timer()
for vector in word_vectors:
    delta = max_vec - len(vector)
    if delta > 0:
        if len(vector) > 0:
            mean = sum(vector) / float(len(vector))
        else:
            failed += 1
            mean = 0
        # append the mean after the vector to make equal sizes
        word_vectors[index] = post_append(vector, mean)
        # append 0 after the vector to make equal sizes
        # word_vectors[index] = post_append(vector)
        # append the mean before the vector to make equal sizes
        # word_vectors[index] = pre_append(vector, mean)
        # append 0 before the vector to make equal sizes
        # word_vectors[index] = pre_append(vector)
    if index < train_count:
        train_tweets.append(word_vectors[index])
    else:
        test_tweets.append(word_vectors[index])
    index += 1
stop = timeit.default_timer()
print("padding")
print("--------------------------------------")
print(stop - start)
print("--------------------------------------")
print(len(word_vectors))
print(len(word_vectors[0]))
print(len(word_vectors[10]))
print(len(train_labels))
print(len(train_tweets))
print(len(test_labels))
print(len(test_tweets))
for name, clf in zip(names, classifiers):
    start = timeit.default_timer()
    clf.fit(np.asarray(train_tweets), np.asarray(train_labels).T)
    score = clf.score(np.asarray(test_tweets), np.asarray(test_labels).T)
    print(name)
    print("<><<><><><><><><><")
    print(score)
    stop = timeit.default_timer()
    print(stop - start)
    print("<><<><><><><><><><")
print("DONE")
print(failed)