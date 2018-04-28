from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans, LDA
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

conf = SparkConf()
conf.setAppName("TwitterStreamApp")

sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

sqlContext = SQLContext(sc)

df1 = sqlContext.read.format("csv").option("header", "true").option("mode", "DROPMALFORMED").load("politics.csv")
df2 = sqlContext.read.format("csv").option("header", "true").option("mode", "DROPMALFORMED").load(
    "a1newCSVFullTweets.csv")
df = df1.unionAll(df2)
# df.show()
# df2.show()

tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
remover = StopWordsRemover(inputCol="tokens", outputCol="stopWordsRemovedTokens")
hashingTF = HashingTF(inputCol="stopWordsRemovedTokens", outputCol="rawFeatures", numFeatures=2000)
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)

kmeans = KMeans(k=2)
lda = LDA(k=2, seed=1, optimizer="em",featuresCol='features')

#pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, kmeans])
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf,lda])
model = pipeline.fit(df)
topics = model.stages[-1].describeTopics()

topics.show(truncate=False)
transformed = model.transform(df)
transformed.sort('topicDistribution').show(20000,truncate=False)
transformed.toPandas().to_csv('ldaresults.csv')



# results = model.transform(df)
# results.cache()

#results.groupBy("prediction").count().show()  # Note "display" is for Databricks; use show() for OSS Apache Spark
#results.filter(results.prediction == 1).show(200,False)
# results.show()


