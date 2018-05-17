import requests
import requests_oauthlib
import json
import os
import socket
import csv
import pickle
import pandas as pd

ACCESS_TOKEN = '168495059-UE3cKbDqaIuKw6ffgNhxBN5zavHWA2dM5Qr4p72Y'
ACCESS_TOKEN_SECRET = '5zbtfijk9gFa8dHECR38SVe33kK6vpQE9V6hkHNyeAOR2'
CONSUMER_KEY = 'K53IN4zgHjktXFFvfvRFaW8bw'
CONSUMER_SECRET = 'gmzJtlb3ZLEoo3OsbzeLJt3fyrGZFihbfHWvIdtjeIeXEeMPan'
auth_worker = requests_oauthlib.OAuth1(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

IP = 'localhost'
PORT = 9999


def stream_tweets():
    url = 'https://stream.twitter.com/1.1/statuses/filter.json'
    query_data = [('language', 'en'), ('locations', '-130,-20,100,50'), ('track', '#')]
    query_url = url + '?' + '&'.join([str(t[0]) + '=' + str(t[1]) for t in query_data])
    response = requests.get(query_url, auth=auth_worker, stream=True)
    print(query_url, response)
    return response
def stream_tweets_in_usa():
    url = 'https://stream.twitter.com/1.1/statuses/filter.json'
    query_data = [('language', 'en'), ('locations', '-136,15,-45,55'), ('track', '#')]
    query_url = url + '?' + '&'.join([str(t[0]) + '=' + str(t[1]) for t in query_data])
    response = requests.get(query_url, auth=auth_worker, stream=True)
    print(query_url, response)
    return response

# fp = open('tweets.json', 'wb+')


def save_tweets_to_file(file, response):
    size = os.path.getsize('./tweets.json')
    while size < 4096:
        i = 1

        for line in response.iter_lines():
            file.write(line + b'\n')
            # file.write(b'\n')
            print(i, size)
            i += 1
    file.close()
    print('wrote tweets: ', i)


def read_json():
    with open('tweets.json') as f:
        content = f.readlines()
    for line in content:
        d = json.loads(line)
        print(d['text'])
        print(line)


def send_tweets_to_spark(http_response, connection):
    for line in http_response.iter_lines():
        try:
            full_tweet = json.loads(line)

            tweet_text = full_tweet['text']
            print('full tweet:', full_tweet)
            print('tweet text:', tweet_text)
            print('-' * 10)
            connection.send(bytearray(str(tweet_text) + '\n', 'utf8'))
        except Exception as e:
            print(e)

def send_tweets_to_spark_with_location(http_response, connection):
    for line in http_response.iter_lines():
        try:
            full_tweet = json.loads(line)
            tweet_text = full_tweet['text']

            #print('full tweet:', full_tweet)
            #print('tweet text:', tweet_text)
            print('-' * 10)
            my_dict = dict([('text', full_tweet['text']), ('coordinates', full_tweet['coordinates'])])
            if my_dict['coordinates'] is not None:
                print(my_dict)
                str_my_dict = json.dumps(my_dict)
                print(str_my_dict)
                connection.send(bytearray(str(str_my_dict) + '\n', 'utf8'))

        except Exception as e:
            print(e)


def read_tweets_forever(response):
    for line in response.iter_lines():
        try:
            full_tweet = json.loads(line)
            tweet_text = full_tweet['text']
            print('full tweet:', full_tweet)
            print('tweet text:', tweet_text)
            print('-' * 10)
            return full_tweet
        except Exception as e:
            print(e)


# send_tweets_to_spark(stream_tweets(), None)
# save_tweets_to_file(fp, stream_tweets())

# read_json()

def start_server(loc = False):
    conn = None
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((IP, PORT))
    s.listen(1)
    print("Waiting tcp conn")
    conn, addr = s.accept()
    print("Connected. Starting getting tweets.")
    resp = stream_tweets()
    if loc:
        send_tweets_to_spark_with_location(resp, conn)
    else:
        send_tweets_to_spark(resp, conn)




def get_tweet_from_id(id):
    url = "https://api.twitter.com/1.1/statuses/show.json?id=" + str(id)
    response = requests.get(url, auth=auth_worker, stream=True)
    for line in response.iter_lines():
        my_full_tweet = json.loads(line)
    return my_full_tweet


wanted_keys = ['text', 'label', 'id', 'created_at', 'truncated', 'user', 'geo', 'coordinates', 'retweet_count',
               'favorite_count', 'lang']


def write_jsons_to_csv(myjson, header=False, label=0):
    with open('a' + str(label) + 'newCSVFullTweets.csv', 'a', encoding="utf-8") as outcsv:
        writer = csv.writer(outcsv)
        if header == True:
            writer.writerow(myjson.keys())
        writer.writerow(myjson.values())


def read_tweets_from_file_by_id(file):
    headerzero = True
    headerone = True
    with open(file, 'r', encoding="utf-8") as incsv:
        reader = csv.DictReader(incsv)
        for row in reader:
            # print(row)
            tweet = get_tweet_from_id(row['id'])
            label = row['label']

            available = True
            for key in tweet.keys():
                if key.startswith('er'):
                    available = False

            if available:
                tweet['label'] = label
                #print(row['id'], tweet['label'], tweet)
                items = tweet.items()
                new_tweet = {}
                for item in items:
                    key = item[0]
                    if key in wanted_keys:
                        new_tweet[key] = item[1]

                if label == '0':
                    write_jsons_to_csv(new_tweet, header=headerzero, label=label)
                    headerzero = False
                if label == '1':
                    write_jsons_to_csv(new_tweet, header=headerone, label=label)
                    headerone = False


def convert_to_csv_from_xlsx(file):
    data_xls = pd.read_excel(file, index_col=None)
    data_xls.to_csv('newDATASET.csv', encoding='utf-8', index=False)


# atweet =  get_tweet_from_id(732113334737207296)

def read_from_csv(file):
    with open(file, 'r', encoding='utf-8') as incsv:
        reader = csv.DictReader(incsv)
        for line in reader:
            print(line['text'], line['label'])


#read_tweets_from_file_by_id('newCSV.csv')
start_server(loc = True)