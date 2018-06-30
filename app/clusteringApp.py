from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.clustering import KMeans, LDA, DistributedLDAModel, KMeansModel
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.clustering import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

def kmeans_from_csv(file="liveTweetsLocation.csv",outfile="liveTweetsLocationKmeans.csv",k=8):
    df = sqlContext.read.format("csv").option("header", "true").option("mode", "DROPMALFORMED").load \
        (file)
    df.show()
    # df2.show()

    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    remover = StopWordsRemover(inputCol="tokens", outputCol="stopWordsRemovedTokens")
    hashingTF = HashingTF(inputCol="stopWordsRemovedTokens", outputCol="rawFeatures", numFeatures= 2 **20)
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    kmeans = KMeans(k=k, seed=1 ,featuresCol='features' ,maxIter=10 ,initMode='random')
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf,  kmeans])
    model = pipeline.fit(df)
    results = model.transform(df)
    results.cache()
    results.groupBy("prediction").count().show()  # Note "display" is for Databricks; use show() for OSS Apache Spark
    # results.filter(results.prediction == 1).show(200,False)
    results.show()
    results.toPandas().to_csv(outfile)
    results.drop("location").drop("tokens").drop("stopWordsRemovedTokens").drop("rawFeatures").drop("features").toPandas().to_csv('prettyPrint.csv')

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
    results = model.transform(df)
    results.cache()
    results.groupBy("prediction").count().show()  # Note "display" is for Databricks; use show() for OSS Apache Spark
    results.filter(results.prediction == 1).show(200,False)
    results.show()
    results.toPandas().to_csv('gmmresultsCanadaAndProductsAndDisastersAndClaritin.csv')

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

def kmeans_from_csv2(file="liveTweetsLocation.csv",k=8):
    df = sqlContext.read.format("csv").option("header", "true").option("mode", "DROPMALFORMED").load \
        (file)
    df.show()
    # df2.show()

    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    remover = StopWordsRemover(inputCol="tokens", outputCol="stopWordsRemovedTokens")
    hashingTF = HashingTF(inputCol="stopWordsRemovedTokens", outputCol="rawFeatures", numFeatures= 2 **20)
    idf = IDF(inputCol="rawFeatures", outputCol="features")


    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])
    model = pipeline.fit(df)
    results = model.transform(df)
    results.cache()
    #results.groupBy("prediction").count().show()  # Note "display" is for Databricks; use show() for OSS Apache Spark
    # results.filter(results.prediction == 1).show(200,False)
    results.show()
    #results.toPandas().to_csv(outfile)
    # Trains a k-means model.
    xaxis = []
    yaxis = []
    for k in range(2,11):
        xaxis.append(k)
        kmeans = KMeans().setK(k).setSeed(1)
        model = kmeans.fit(results)

        # Evaluate clustering by computing Within Set Sum of Squared Errors.
        wssse = model.computeCost(results)
        yaxis.append(wssse)
        print("Within   Sum of Squared Errors  for k= "+ str(k) +"is " + str(wssse))
    plt.plot(xaxis,yaxis)
    plt.show()


def kmeansresults2():
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
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])
    model = pipeline.fit(df)
    results = model.transform(df)
    results.cache()

    # results.filter(results.prediction == 1).show(200,False)
    results.show()

    xaxis = []
    yaxis = []
    for k in range(2, 19):
        try:
            xaxis.append(k)

            kmeans = KMeans().setK(k).setSeed(1)
            model = kmeans.fit(results)
            print('fitted!')
            # Evaluate clustering by computing Within Set Sum of Squared Errors.
            wssse = model.computeCost(results)
            yaxis.append(wssse)
            print("Within Set Sum of Squared Errors  for k= " + str(k) + "is " + str(wssse))
        except Exception as e:
            print(e)            
    if len(xaxis) != len(yaxis):
        length = min(len(xaxis),len(yaxis))
        xaxis = xaxis[:length]
        yaxis = yaxis[:length]
    plt.plot(xaxis, yaxis)
    plt.show()





conf = 0

sc = 0
sqlContext = 0

if __name__ == '__main__':
    conf = SparkConf()
    conf.setAppName("TwitterStreamApp")
    conf = (conf.setMaster('local[*]')
           .set('spark.executor.memory', '4G')
           .set('spark.driver.memory', '4G')
           .set('spark.driver.maxResultSize', '4G'))

    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    sqlContext = SQLContext(sc)
    #kmeansresults()
    #ldaresults()
    #gmmresults()
    #kmeans_with_loading()
    #kmeans_from_csv2("..\liveTweetsLocation.csv","liveTweetsLocationKmeans")
    #kmeansresults2()




