from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import LogisticRegressionWithSGD, LogisticRegressionModel
from pyspark.mllib.feature import HashingTF
from pyspark.streaming import StreamingContext
import sys
import requests
from pyspark.sql import Row, SQLContext
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer(strip_handles=True,reduce_len=True)
tf = HashingTF(numFeatures=2 ** 18)

def print_rdd_with_prediction(rdd):
    print("RECORD: ")
    tweets = rdd.collect()
    for tweet in tweets:
        words = tokenizer.tokenize(tweet)
        print('one tweet: ', words)
        hashed = tf.transform(words)
        print('transformed',hashed )
        prediction = model.predict(hashed)
        print('prediction ', prediction)
    print(20 * "-")


conf = SparkConf()
conf.setAppName("TwitterStreamApp")

sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
model = LogisticRegressionModel.load(sc,"classifierModel")
# create the Streaming Context from the above spark context with interval size 2 seconds
ssc = StreamingContext(sc, 2)

dataStream = ssc.socketTextStream("localhost", 9999)

# split tweets
tweets = dataStream.flatMap(lambda line: line.split("\n"))
tweets.foreachRDD(print_rdd_with_prediction)
# tweets.pprint()

ssc.start()

ssc.awaitTermination()
