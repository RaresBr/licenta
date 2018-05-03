from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.clustering import KMeans, LDA, DistributedLDAModel, KMeansModel
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
from pyspark.ml.clustering import GaussianMixture
import numpy as np
def kmeans_from_csv(file):
    df = sqlContext.read.format("csv").option("header", "true").option("mode", "DROPMALFORMED").load \
        (file)
    df.show()
    # df2.show()

    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    remover = StopWordsRemover(inputCol="tokens", outputCol="stopWordsRemovedTokens")
    hashingTF = HashingTF(inputCol="stopWordsRemovedTokens", outputCol="rawFeatures", numFeatures= 2 **20)
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    kmeans = KMeans(k=8, seed=1 ,featuresCol='features' ,maxIter=10 ,initMode='random')
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf,  kmeans])
    model = pipeline.fit(df)
    results = model.transform(df)
    results.cache()
    results.groupBy("prediction").count().show()  # Note "display" is for Databricks; use show() for OSS Apache Spark
    # results.filter(results.prediction == 1).show(200,False)
    results.show()
    results.toPandas().to_csv('liveTweetsKMeans.csv')

def kmeans_with_pipeline(df,pipeline):
    model = pipeline.fit(df)
    results = model.transform(df)
    results.cache()
    results.groupBy("prediction").count().show()  # Note "display" is for Databricks; use show() for OSS Apache Spark
    # results.filter(results.prediction == 1).show(200,False)
    results.show()
    results.toPandas().to_csv('kmeansresultsLiveTweets.csv')

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
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf,  kmeans])
    pipeline.save("KMeansPipeline")
    model = pipeline.fit(df)
    results = model.transform(df)
    results.cache()
    results.groupBy("prediction").count().show()  # Note "display" is for Databricks; use show() for OSS Apache Spark
    # results.filter(results.prediction == 1).show(200,False)
    results.show()
    results.toPandas().to_csv('kmeansresultsCanadaAndProductsAndDisastersAndClaritin.csv')
    model.stages[-1].save("KMeansModel")


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

    lda = LDA(k=8, seed=1, optimizer="em", featuresCol='features')

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
#out of memory gmm
def gmmresults():
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
    hashingTF = HashingTF(inputCol="stopWordsRemovedTokens", outputCol="rawFeatures", numFeatures= 20000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    gmm = GaussianMixture(k=8 ,featuresCol='features')
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf,gmm])
    model = pipeline.fit(df)
    # results = model.transform(df)
    # results.cache()
    # results.groupBy("prediction").count().show()  # Note "display" is for Databricks; use show() for OSS Apache Spark
    # # results.filter(results.prediction == 1).show(200,False)
    # results.show()
    # results.toPandas().to_csv('gmmresultsCanadaAndProductsAndDisastersAndClaritin.csv')

def kmeans_with_loading():
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
    clusteringPipeline = Pipeline.load('KMeansPipeline')
    model = clusteringPipeline.fit(df)
    transf = model.transform(df)
    transf.show(200, False)




if __name__ == '__main__':
    conf = SparkConf()
    conf.setAppName("TwitterStreamApp")

    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    sqlContext = SQLContext(sc)
    #kmeansresults()
    #ldaresults()
    #gmmresults()
    #kmeans_with_loading()
    kmeans_from_csv("..\liveTweets.csv")




