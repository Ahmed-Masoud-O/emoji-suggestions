import preprocessor as p
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
import re
import timeit


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

tokenizer = TweetTokenizer()
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')
word_vectors = []
length_vector = []
index = 0
start = timeit.default_timer()
text_file = open("cleanedTweets.txt", "w")
for tweets in open('tweets.txt', encoding="utf-8"):
    p.set_options(p.OPT.HASHTAG)
    parsed_tweet = p.parse(tweets)
    hashtags = parsed_tweet.hashtags
    splits = []
    if hashtags:
        for hashtag in hashtags:
            split_hashtags(hashtag)
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.NUMBER, p.OPT.HASHTAG)
    stop_words = set(stopwords.words("english"))
    extra_stop_words = ['.', '|', '+', '~', '✓', '︎', '“', "'", '—', '⠀', '-', ',', '•', '・', '_', '!', '&', ')', '(', '…', '️', ' ',  '...', '"', '/', '?', '', '..', ':']
    for symbol in extra_stop_words:
        stop_words.add(symbol)
    cleaned_tweet = p.clean(tweets)
    tokenized_tweet = tokenizer.tokenize(cleaned_tweet)
    filtered_tweet = [w for w in tokenized_tweet if not w.lower() in stop_words and len(w) > 0]
    for split in splits:
        no_numbers = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", split)
        if no_numbers not in stop_words and len(no_numbers) > 0:
            filtered_tweet.append(no_numbers)
    for idx, word in enumerate(filtered_tweet):
        filtered_tweet[idx] = lemmatizer.lemmatize(word)
        filtered_tweet[idx] = stemmer.stem(word)
    for word in filtered_tweet:
        try:
            text_file.write(word)
            text_file.write(' ')
        except:
            continue
    text_file.write("\n")
stop = timeit.default_timer()
print("cleaning")
print("--------------------------------------")
print(stop - start)
print("--------------------------------------")