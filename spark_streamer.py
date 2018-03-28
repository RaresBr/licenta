from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
import sys
import requests
from pyspark.sql import Row, SQLContext
def print_rdd(rdd):
    print("RECORD: ")
    print(rdd.collect())
    print(20*"-")
conf = SparkConf()
conf.setAppName("TwitterStreamApp")

sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
# create the Streaming Context from the above spark context with interval size 2 seconds
ssc = StreamingContext(sc, 2)

dataStream = ssc.socketTextStream("localhost", 9999)

# split tweets
tweets = dataStream.flatMap(lambda line: line.split("\n"))
tweets.foreachRDD(print_rdd)
#tweets.pprint()

ssc.start()

ssc.awaitTermination()
