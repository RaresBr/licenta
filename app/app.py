import easygui as g
from pyspark import SparkConf, SparkContext, SQLContext

import spark_streamer as streamer
import os
from threading import Thread
import time
class WaitMsg(Thread):
    def __init__(self):
        Thread.__init__(self)
    def run(self):
        g.msgbox("Waiting for tweets to be collected.", title=title)
conf = SparkConf()
conf.setAppName("TwitterStreamApp")
conf = (conf.setMaster('local[*]')
       .set('spark.executor.memory', '4G')
       .set('spark.driver.memory', '4G')
       .set('spark.driver.maxResultSize', '4G'))

sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

sqlContext = SQLContext(sc)

streamer.conf = conf
streamer.sc=sc
streamer.sqlContext=sqlContext

title = "Twitter Story Maker"
tweetCount = g.integerbox("Number of tweets to be collected:", title=title,upperbound=1000)
streamer.setTweets(tweetCount)
print(streamer.nrOfTweets)
waitingMsg = WaitMsg()
waitingMsg.start()
try:
    streamer.start()
    print('x')
except Exception as e:
    print(e)
os.startfile('liveTweetsLocation.csv')
import clusteringApp as clustering
clustering.conf = conf
clustering.sc  =sc
clustering.sqlContext=sqlContext
clustering.kmeans_from_csv2()
k = g.integerbox("Insert k", title=title)
clustering.kmeans_from_csv(k=k)

import mapping as m
os.startfile('prettyPrint.csv')
term = g.enterbox("Insert wanted term")
cluster =m.getMaxCluster(term,k)
m.mapping(cluster,term)
os.startfile('my_mapStamen.html')