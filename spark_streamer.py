from pyspark import SparkContext, SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeansModel
from pyspark.mllib.classification import LogisticRegressionWithSGD, LogisticRegressionModel
from pyspark.mllib.feature import HashingTF
from pyspark.streaming import StreamingContext
from ds_clust import clustering
from pyspark.sql import SQLContext
import sys
import requests
from pyspark.sql import Row, SQLContext
from pyspark.sql.types import StringType, StructType, StructField

from nltk.tokenize import TweetTokenizer
import csv
import json



tokenizer = TweetTokenizer(strip_handles=True,reduce_len=True)
tf = HashingTF(numFeatures=2 ** 18)
schema = StructType([StructField("value", StringType(), True)])

def print_rdd_with_prediction(rdd):
    print("RECORD: ")
    global  bigDF
    if not rdd.isEmpty():

        #bigDF = bigDF.unionAll(sqlContext.createDataFrame(rdd,StringType()))
        #print(bigDF.count())
        tweets = rdd.collect()
        with open('liveTweets.csv', 'a',encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for tweet in tweets:


                words = list(map(lambda x : x.lower(), tokenizer.tokenize(tweet)))
                print('one tweet: ', words)
                hashed = tf.transform(words)
                print('transformed',hashed )
                prediction = model.predict(hashed)
                print('prediction ', prediction)
                print('\n')
                writer.writerow([tweet,prediction])
    print(20 * "-")

def print_rdd(rdd):
    print("RECORD: ")
    if not rdd.isEmpty():
        tweets = rdd.collect()
        for tweet in tweets:
            tweet = json.loads(tweet)
            words = list(map(lambda x: x.lower(), tokenizer.tokenize(tweet['text'])))
            print('one tweet: ', words)
            print('location', tweet['coordinates']['coordinates'])
            hashed = tf.transform(words)
            print('transformed', hashed)
            print('\n')
    print(20 * "-")

def print_with_location_rdd_with_prediction(rdd):
    print("RECORD: ")
    global  bigDF
    if not rdd.isEmpty():

        #bigDF = bigDF.unionAll(sqlContext.createDataFrame(rdd,StringType()))
        #print(bigDF.count())
        tweets = rdd.collect()
        with open('liveTweetsLocation.csv', 'a',encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for tweet in tweets:
                tweet = json.loads(tweet)
                words = list(map(lambda x: x.lower(), tokenizer.tokenize(tweet['text'])))
                print('one tweet: ', words)
                hashed = tf.transform(words)
                print('transformed',hashed )
                prediction = model.predict(hashed)
                print('prediction ', prediction)
                coord = tweet['coordinates']['coordinates']
                print('location', coord)
                print('\n')
                writer.writerow([tweet['text'],prediction,coord])
    print(20 * "-")

conf = SparkConf()
conf.setAppName("TwitterStreamApp")

sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)

model = LogisticRegressionModel.load(sc,"classifierModelPLOS")
clusteringPipeline = Pipeline.load('.\ds_clust\KMeansPipeline')
bigDF = sqlContext.createDataFrame("S",StringType())
# create the Streaming Context from the above spark context with interval size 2 seconds
ssc = StreamingContext(sc, 2)

dataStream = ssc.socketTextStream("localhost", 9999)

# split tweets
tweets = dataStream.flatMap(lambda line: line.split("\n"))
#tweets.foreachRDD(print_rdd_with_prediction)
#tweets.foreachRDD(print_rdd)
tweets.foreachRDD(print_with_location_rdd_with_prediction)
# tweets.pprint()
notVisited = True

ssc.start()
# while(True):
#     if bigDF.count() >= 5 and notVisited:
#         notVisited = False
#         newBigDf = bigDF.withColumnRenamed('value','text')
#         newBigDf.show(200, False)
#         #clustering.kmeans_with_pipeline(newBigDf,clusteringPipeline)
ssc.awaitTermination()



