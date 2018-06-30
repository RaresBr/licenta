from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD, NaiveBayes, LogisticRegressionWithLBFGS, SVMWithSGD
from pyspark.mllib.tree import DecisionTree, RandomForest


def score(model):
    features = test_data.map(lambda x: x.features)
    predictions = model.predict(features)

    labels = test_data.map(lambda x: x.label)
    label_with_prediction = labels.zip(predictions)

    elements_gotten_right = label_with_prediction.filter(lambda x: x[0] == x[1]).count()
    all_elements = test_data.count()

    accuracy = elements_gotten_right / float(all_elements)
    return accuracy

def score_new(model):
    features = []
    for element in test_data:
        features.append(element.features)

    predictions = model.predict(features)

    labels = []
    for element in test_data:
        labels.append(element.label)

    labels_with_predictions = zip(labels, predictions)

    elements_gotten_right = []
    for element in labels_with_predictions:
        if element[0] == element[1]:
            elements_gotten_right.append(element)
    return len(elements_gotten_right) / float(len(test_data))


conf = SparkConf()
conf.setAppName("TweetClassifier")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

fake = sc.textFile("a1newCSVFullTweets_cleanLOWERCASE.txt")
real = sc.textFile("a0newCSVFullTweets_cleanLOWERCASE.txt")

fake_words = fake.map(lambda sentence: sentence.split())
real_words = real.map(lambda sentence: sentence.split())

print(fake_words.take(1))
print(real_words.take(1))

tf = HashingTF(numFeatures=2 ** 18)
fake_features = tf.transform(fake_words)
real_features = tf.transform(real_words)

print(fake_features.take(1))
print(real_features.take(1))

# label each element; either fake or real sentence
fake_samples = fake_features.map(lambda features: LabeledPoint(1, features))
real_samples = real_features.map(lambda features: LabeledPoint(0, features))

print(fake_samples.take(1))
print(real_samples.take(1))

samples = fake_samples.union(real_samples)
[training_data, test_data] = samples.randomSplit([0.8, 0.2])
training_data.cache()
test_data.cache()

algorithm = LogisticRegressionWithSGD()
model = algorithm.train(training_data)
print('logistic regression sgd:', score(model))

algorithm = LogisticRegressionWithLBFGS()
model = algorithm.train(training_data)
print('logistic regression with lbfgs:', score(model))

# algorithm = DecisionTree()
# model = algorithm.trainClassifier(training_data, numClasses=2,categoricalFeaturesInfo={})
# print('decision tree: ',score(model))
#
# algorithm = RandomForest()
# model = algorithm.trainClassifier(training_data,numClasses=2,categoricalFeaturesInfo={},numTrees=16)
# print('random forest: ',score(model))

algorithm = NaiveBayes()
model = algorithm.train(training_data)
print('naive bayes: ', score(model))

algorithm = SVMWithSGD()
model = algorithm.train(training_data, iterations=10)
print('svm with sgd: ', score(model))
# model.save(sc,"classifierModelPLOS")
if __name__ == "__main__":
    pass



