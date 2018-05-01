from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans, LDA, DistributedLDAModel
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
import numpy as np
conf = SparkConf()
conf.setAppName("TwitterStreamApp")

sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

sqlContext = SQLContext(sc)


def kmeansresults():
    df1 = sqlContext.read.format("csv").option("header", "true").option("mode", "DROPMALFORMED").load \
        ("canadatweets.csv")
    df2 = sqlContext.read.format("csv").option("header", "true").option("mode", "DROPMALFORMED").load(
        "products.csv")
    df3 = sqlContext.read.format("csv").option("header", "true").option("mode", "DROPMALFORMED").load(
       "products.csv")
    df4 = sqlContext.read.format("csv").option("header", "true").option("mode", "DROPMALFORMED").load(
        "claritin.csv")
    df = df1.unionAll(df2)
    df = df.unionAll(df3)
    df = df.unionAll(df4)
    df.show()
    # df2.show()

    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    remover = StopWordsRemover(inputCol="tokens", outputCol="stopWordsRemovedTokens")
    hashingTF = HashingTF(inputCol="stopWordsRemovedTokens", outputCol="rawFeatures", numFeatures= 2 **20)
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    kmeans = KMeans(k=8, seed=1 ,featuresCol='rawFeatures' ,maxIter=10 ,initMode='random')
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF,  kmeans])
    model = pipeline.fit(df)
    results = model.transform(df)
    results.cache()
    results.groupBy("prediction").count().show()  # Note "display" is for Databricks; use show() for OSS Apache Spark
    # results.filter(results.prediction == 1).show(200,False)
    results.show()
    results.toPandas().to_csv('kmeansresultsCanadaAndProductsAndDisastersAndClaritin.csv')


def ldaresults():
    df1 = sqlContext.read.format("csv").option("header", "true").option("mode", "DROPMALFORMED").load \
        ("canadatweets.csv")
    df2 = sqlContext.read.format("csv").option("header", "true").option("mode", "DROPMALFORMED").load(
        "products.csv")
    df3 = sqlContext.read.format("csv").option("header", "true").option("mode", "DROPMALFORMED").load(
       "products.csv")
    df4 = sqlContext.read.format("csv").option("header", "true").option("mode", "DROPMALFORMED").load(
        "claritin.csv")
    df = df1.unionAll(df2)
    df = df.unionAll(df3)
    df = df.unionAll(df4)
    df.show()
    # df2.show()

    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    remover = StopWordsRemover(inputCol="tokens", outputCol="stopWordsRemovedTokens")
    hashingTF = HashingTF(inputCol="stopWordsRemovedTokens", outputCol="rawFeatures", numFeatures= 2 **18)
    idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)

    kmeans = KMeans(k=20, seed=1)
    lda = LDA(k=4, seed=1, optimizer="em", featuresCol='features')

    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lda])
    model = pipeline.fit(df)
    topics = model.stages[-1].describeTopics()

    # topics.show(truncate=False)
    transformed = model.transform(df)

    # transformed.sort('topicDistribution').show(20000,truncate=False)
    # transformed.toPandas().to_csv('ldaresultsCanadaAndProductsAndDisastersAndClaritin.csv')
    transformed.rdd.map(lambda row: (row['text'] ,row['features'] ,row['topicDistribution'],int(np.argmax(np.asarray([float(x) for x in row['topicDistribution']])))))\
        .toDF()\
        .toPandas().to_csv('ldaresultsCanadaAndProductsAndDisastersAndClaritin.csv')


    # transformed.withColumn('TopicNumber',lit(np.argmax(np.asarray(list(transformed.topicDistribution))))).show(5,truncate=False)
    # model.stages[-1].save("LDAModel")


# kmeansresults()
ldaresults()




