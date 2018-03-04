import requests
import requests_oauthlib
import json
import os
import socket

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
            connection.send(bytearray(str(full_tweet) + '\n', 'utf8'))
        except Exception as e:
            print(e)


#send_tweets_to_spark(stream_tweets(), None)
# save_tweets_to_file(fp, stream_tweets())

#read_json()
conn = None
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((IP, PORT))
s.listen(1)
print("Waiting for TCP connection...")
conn, addr = s.accept()
print("Connected... Starting getting tweets.")
resp = stream_tweets()
send_tweets_to_spark(resp, conn)
