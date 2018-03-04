from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
import sys
import requests
from pyspark.sql import Row,SQLContext

conf = SparkConf()
conf.setAppName("TwitterStreamApp")
# create spark context with the above configuration
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
# create the Streaming Context from the above spark context with interval size 2 seconds
ssc = StreamingContext(sc, 1)


dataStream = ssc.socketTextStream("localhost",9999)

# split tweets
tweets = dataStream.flatMap(lambda line: line.split("\n"))
tweets.pprint(200)

ssc.start()

ssc.awaitTermination()

