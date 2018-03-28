# licenta
Work in progress. 
A twitter spam/troll/propaganda classifier. At the moment it can predict whether a tweet is fake or not. 
I used a dataset with 2016's russian propaganda movement on social media (namely twitter).

Next step would be to clusterize tweets that have been deemed as not spamish and thus provide a useful tool for journalists, with relevant stories from the last, say, 24 hours.
How to run:
-run streaming_twitter_api_calls.py
-run spark_streamer.py
This will get real time tweets to Spark.

-run classifier.py
This uses LinearRegression to classify test data. Accuracy is about 93%. Will work on it more. 

Final project for FII. 

