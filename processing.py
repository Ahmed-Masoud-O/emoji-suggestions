from nltk.tokenize import TweetTokenizer
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
from helper import *
import matplotlib.pyplot as plt
plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


# def post_append(temp_vector, number=0.0):
#     extra_vector = [number] * delta
#     # for i in range(delta):
#     #     temp_vector.append(number)
#     return temp_vector + extra_vector
#
#
# def pre_append(temp_vector, number=0.0):
#     zeros_vector = [number] * delta
#     temp_vector = zeros_vector + temp_vector
#     return temp_vector
def draw_accuracy():
    objects = names
    y_pos = np.arange(len(objects))
    performance = scores

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('accuracy')
    plt.title('classifiers')
    plt.show()


def draw_f_measure():
    objects = names
    y_pos = np.arange(len(objects))
    performance = f_measures

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('f-measure')
    plt.title('classifiers')
    plt.show()


def draw_precision():
    objects = names
    y_pos = np.arange(len(objects))
    performance = precisions

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('precision')
    plt.title('classifiers')
    plt.show()


def draw_recall():
    objects = names
    y_pos = np.arange(len(objects))
    performance = recalls

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('recall')
    plt.title('classifiers')
    plt.show()

names = ["Nearest Neighbors",
         "Decision Tree",
         "Random Forest",
         "Neural Net",
         "AdaBoost",
         "Naive Bayes",
         "QDA",
         ]

classifiers = [
    KNeighborsClassifier(15),
    DecisionTreeClassifier(max_depth=20),
    RandomForestClassifier(max_depth=20, n_estimators=10, max_features=1),
    MLPClassifier(alpha=0.5),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),

]
tokenizer = TweetTokenizer()
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
train_count = 350000
for tweets in open('cleanedTweets.txt', encoding="utf-8"):
    index += 1
    cleaned_tweet = tweets
    tokenized_tweet = tokenizer.tokenize(cleaned_tweet)
    filtered_tweet = tokenized_tweet
    tweet_vector = []
    for idx, word in enumerate(filtered_tweet):
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
for vector, labels in zip(word_vectors, my_labels):
    # delta = max_vec - len(vector)
    # if delta > 0:
    #     if len(vector) > 0:
    #         mean = sum(vector) / float(len(vector))
    #     else:
    #         failed += 1
    #         mean = 0
    #     # append the mean after the vector to make equal sizes
    #     word_vectors[index] = post_append(vector, mean)
    #     # append 0 after the vector to make equal sizes
    #     # word_vectors[index] = post_append(vector)
    #     # append the mean before the vector to make equal sizes
    #     # word_vectors[index] = pre_append(vector, mean)
    #     # append 0 before the vector to make equal sizes
    #     # word_vectors[index] = pre_append(vector)
    if len(vector) > 0:
        simplified_word_vector = [min(vector), max(vector)]
        if index < train_count:
            train_tweets.append(simplified_word_vector)
        else:
            test_tweets.append(simplified_word_vector)
        if index < train_count:
            train_labels.append(labels.strip())
        else:
            test_labels.append(labels.strip())
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
f_measures = []
scores = []
precisions = []
recalls = []
for name, clf in zip(names, classifiers):
    start = timeit.default_timer()
    clf.fit(np.asarray(train_tweets), np.asarray(train_labels))
    score = clf.score(np.asarray(test_tweets), np.asarray(test_labels))
    predictions = clf.predict(np.asarray(test_tweets))
    print(name)
    print("<><<><><><><><><><")
    print("f-measure")
    print("-------------")
    current_measure = f_measure(predictions, test_labels)
    f_measures.append(current_measure[0])
    precisions.append(current_measure[1])
    recalls.append(current_measure[2])
    print(current_measure)
    print("entropy")
    print("-------------")
    print(conditional_entropy(predictions, test_labels))
    print("accuracy")
    print("-------------")
    print(score)
    scores.append(score)
    stop = timeit.default_timer()
    print("time elapsed")
    print("-------------")
    print(stop - start)
    print("<><<><><><><><><><")
print("DONE")
draw_accuracy()
draw_f_measure()
draw_precision()
draw_recall()

