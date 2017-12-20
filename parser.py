import preprocessor as p
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, words, brown
from nltk.tokenize import TweetTokenizer
# import ssl
from nltk.stem.snowball import SnowballStemmer
import re
import fasttext
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import timeit

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", #"Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]


def split_numbers(value):
    return re.match(r"([a-z]+)([0-9]+)", value, re.I)


def split_upper(value):
    return re.sub(r'([A-Z])', r' \1', value)


def split_underscores(value):
    return value.split("_")


def split_hashtags(hashtag):
    split_on_underscores = split_underscores(hashtag.match[1:])
    for underscore_split in split_on_underscores:
        split_on_upper = split_upper(underscore_split)
        for upper_split in split_on_upper.split(" "):
            split_on_numbers = split_numbers(upper_split)
            if split_on_numbers:
                split_on_numbers = split_numbers(upper_split).groups()
                splits.append(split_on_numbers[0])
            else:
                splits.append(upper_split)


def post_append(temp_vector, number=0.0):
    extra_vector = [number] * delta
    # for i in range(delta):
    #     temp_vector.append(number)
    return temp_vector + extra_vector


def pre_append(temp_vector, number=0.0):
    zeros_vector = [number] * delta
    temp_vector = zeros_vector + temp_vector
    return temp_vector

# utf_emoji = [
#      b"\xe2\x9d\xa4", b"\xf0\x9f\x98\x8d", b"\xf0\x9f\x98\x82", b"\xf0\x9f\x92\x95", b"\xf0\x9f\x94\xa5",
#      b"\xf0\x9f\x98\x8a", b"\xf0\x9f\x98\x8e", b"\xe2\x9c\xa8", b"\xf0\x9f\x92\x99", b"\xf0\x9f\x98\x98",
#      b"\xf0\x9f\x93\xb7", b"\xf0\x9f\x87\xba\xf0\x9f\x87\xb8", b"\xe2\x98\x80", b"\xf0\x9f\x92\x9c",
#      b"\xf0\x9f\x98\x89", b"\xf0\x9f\x92\xaf", b"\xf0\x9f\x98\x81", b"\xf0\x9f\x8e\x84", b"\xf0\x9f\x93\xb8",
#      b"\xf0\x9f\x98\x9c"
# ]
# emoji_map = {b"\xe2\x9d\xa4": "_red_heart_", b"\xf0\x9f\x98\x8d": "_smiling_face_with_hearteyes_",
#           b"\xf0\x9f\x98\x82": "_face_with_tears_of_joy_", b"\xf0\x9f\x92\x95": "_two_hearts_",
#           b"\xf0\x9f\x94\xa5": "_fire_", b"\xf0\x9f\x98\x8a": "_smiling_face_with_smiling_eyes_",
#           b"\xf0\x9f\x98\x8e": "_smiling_face_with_sunglasses_", b"\xe2\x9c\xa8": "_sparkles_",
#           b"\xf0\x9f\x92\x99": "_blue_heart_", b"\xf0\x9f\x98\x98": "_face_blowing_a_kiss_",
#           b"\xf0\x9f\x93\xb7": "_camera_", b"\xf0\x9f\x87\xba\xf0\x9f\x87\xb8": "_United_States_",
#           b"\xe2\x98\x80": "_sun_", b"\xf0\x9f\x92\x9c": "_purple_heart_", b"\xf0\x9f\x98\x89": "_winking_face_",
#           b"\xf0\x9f\x92\xaf": "_hundred_points_", b"\xf0\x9f\x98\x81": "_beaming_face_with_smiling_eyes_",
#           b"\xf0\x9f\x8e\x84": "_Christmas_tree_", b"\xf0\x9f\x93\xb8": "_camera_with_flash_",
#           b"\xf0\x9f\x98\x9c": "_winking_face_with_tongue_"}

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
# start = timeit.default_timer()
# model = fasttext.skipgram('Output.txt', 'model')
# stop = timeit.default_timer()
# print("model created")
# print("-----------------------")
# print(stop - start)
# print("-----------------------")
# text_file = open("Output.txt", "w")
limit = 1000
train_count = 700
model = fasttext.load_model('model.bin')
start = timeit.default_timer()
for tweets, labels in zip(open('Output.txt', encoding="utf-8"), my_labels):
    if index == limit:
        break
    if index < train_count:
        train_labels.append(labels.strip())
    else:
        test_labels.append(labels.strip())
    index += 1
    # print(tweets)
    # p.set_options(p.OPT.HASHTAG)
    # parsed_tweet = p.parse(tweets)
    # hashtags = parsed_tweet.hashtags
    # splits = []
    # if hashtags:
    #     for hashtag in hashtags:
    #         split_hashtags(hashtag)
    # p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.NUMBER, p.OPT.HASHTAG)
    # stop_words = set(stopwords.words("english"))
    # extra_stop_words = ['.', '|', '+', '~', '✓', '︎', '“', "'", '—', '⠀', '-', ',', '•', '・', '_', '!', '&', ')', '(', '…', '️', ' ',  '...', '"', '/', '?', '', '..', ':']
    # for symbol in extra_stop_words:
    #     stop_words.add(symbol)
    # cleaned_tweet = p.clean(tweets)
    cleaned_tweet = tweets
    tokenized_tweet = tokenizer.tokenize(cleaned_tweet)
    filtered_tweet = tokenized_tweet
    # filtered_tweet = [w for w in tokenized_tweet if not w.lower() in stop_words and len(w) > 0]
    # for split in splits:
    #     filtered_tweet.append(split)
    # print(filtered_tweet)
    tweet_vector = []
    for idx, word in enumerate(filtered_tweet):
        # filtered_tweet[idx] = lemmatizer.lemmatize(word)
        # filtered_tweet[idx] = stemmer.stem(word)
        # if filtered_tweet[idx].encode() in utf_emoji:
        #     filtered_tweet[idx] = emoji_map[filtered_tweet[idx].encode()]
        tweet_vector = tweet_vector + model[word]
    # print(filtered_tweet)
    length_vector.append(len(tweet_vector))

    # for word in filtered_tweet:
    #     try:
    #         text_file.write(word)
    #         text_file.write(' ')
    #     except:
    #         continue
    # text_file.write("\n")
    word_vectors.append(tweet_vector)
max_vec = max(length_vector)
index = 0
stop = timeit.default_timer()
print("cleaning, tokenizing and vectorization")
print("--------------------------------------")
print(stop - start)
print("--------------------------------------")
# np.save('word_vectors', np.asarray(word_vectors))
# np.save('max_vector', np.asarray(max_vec))
# np.save('train_labels', np.asarray(train_labels))
# np.save('test_labels', np.asarray(test_labels))
# word_vectors = np.load('word_vectors.npy')
# max_vec = np.load('max_vector.npy')
# train_labels = np.load('train_labels.npy')
# test_labels = np.load('test_labels.npy')
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

