import fasttext
import timeit

start = timeit.default_timer()
model = fasttext.skipgram('cleanedTweets.txt', 'model')
stop = timeit.default_timer()
print("model created")
print("-----------------------")
print(stop - start)
print("-----------------------")
