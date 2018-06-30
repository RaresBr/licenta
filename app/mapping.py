import folium
from pyspark import SparkConf, SparkContext, SQLContext
import csv


def getMaxCluster(keyword="I'm"):
    clusters_and_count = {}
    for i in range(0, 8):
        clusters_and_count[i] = 0
    with open('liveTweetsLocationKmeans.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(row['text'], row['location'], row['prediction'])
            if keyword in row['text']:
                clusters_and_count[int(row['prediction'])] += 1
    for k, v in clusters_and_count.items():
        print(k, v)
    max_cluster = max(clusters_and_count, key=clusters_and_count.get)
    print(max_cluster)
    return max_cluster


def mapping(max_cluster,keyword="I'm"):
    my_map = folium.Map(location=[45.5236, -122.6750],tiles='Stamen Toner',)
    with open('liveTweetsLocationKmeans.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            location = list(reversed(eval(row['location'])))
            print(row['text'])
            text = row['text']
            print(location)
            folium.Marker(location,
                          popup=folium.Popup(text,parse_html=True)
                          ).add_to(my_map)
            if int(row['prediction']) == max_cluster:
                if keyword in row['text']:
                    folium.Marker(list(reversed(eval(row['location']))), popup=folium.Popup(row['text'] + ' {Cluster:' + str(row['prediction']) + '}',parse_html=True)
                                  ,icon =  folium.Icon(color='red')).add_to(my_map)
                else:
                    folium.Marker(list(reversed(eval(row['location']))),
                                  popup=folium.Popup(row['text'] + ' {Cluster:' + str(row['prediction']) + '}',
                                                     parse_html=True)
                                  , icon=folium.Icon(color='orange')).add_to(my_map)
            else:
                folium.Marker(list(reversed(eval(row['location']))),
                              popup=folium.Popup(row['text'] + ' {Cluster:' + str(row['prediction']) + '}',
                                                 parse_html=True)
                              , icon=folium.Icon(color='blue')).add_to(my_map)

    my_map.save('my_mapStamen.html')


if __name__ == '__main__':
    mapping(3)
