import pandas as pd
import numpy as np
import sys
import geopandas
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
#https://stackoverflow.com/questions/4971269/how-to-pick-a-new-color-for-each-plotted-line-within-a-figure-in-matplotlib
from matplotlib.pyplot import cm
import geopy.distance
from math import cos, asin, sqrt, pi



def distanceToNearestCluster(point_lon,point_lat,centroids):
	minDistance = 1000
	for i in range(33):
		if(minDistance > distance( centroids.iloc[i]['latitude'], centroids.iloc[i]['longitude'], point_lat, point_lon)):
			minDistance = distance( centroids.iloc[i]['latitude'], centroids.iloc[i]['longitude'], point_lat, point_lon)
	return minDistance

### Thanks to : https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
def distance(lat1, lon1, lat2, lon2):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 12742 * asin(sqrt(a)) #2*R*asin...


def main():
	data = pd.read_json(sys.argv[1],lines = True)
	#import the bus stop data from translink
	stops = pd.read_csv('stops.txt', sep=',')
	stops = stops.loc[(stops['stop_lon'] > -123.5) & (stops['stop_lon'] < -122) & (stops['stop_lat'] > 49) & (stops['stop_lat'] < 49.5)]

	listings = pd.read_csv('listings.csv')
	#https://stackoverflow.com/questions/32464280/converting-currency-with-to-numbers-in-python-pandas
	listings['price_float'] = listings['price'].replace('[\$,]', '', regex=True).astype(float)
	listings = listings.loc[(listings['longitude'] > -123.5) & (listings['longitude'] < -122) & \
	(listings['latitude'] > 49) & (listings['latitude'] < 49.5) & \
	(listings['bedrooms'] == 1) & (listings['price_float'] < 200)
	]
	print(listings)

	X = np.array(stops[['stop_lon','stop_lat']])
	model = DBSCAN(eps=20/6371, min_samples=15, algorithm='ball_tree', metric='haversine').fit(X)
	cluster_point = model.labels_

	print(cluster_point)
	stops_labels = pd.DataFrame(X)
	stops_labels['label'] = cluster_point
	print(stops_labels.groupby('label').count())



	map_van = plt.imread('map.png')
	Boundary = (-123.5,-122,49,49.5)
	fig = plt.figure(figsize = (10,8))
	fig, ax = plt.subplots(figsize = (10,8))
	#ax.scatter(stops['stop_lon'],stops['stop_lat'], zorder=1, alpha= 0.2, c='b', s=5)
	'''
	plotData = stops_labels.loc[stops_labels['label'] == 2]
	plotData = plotData.set_axis(['lon', 'lat', 'label'], axis=1, inplace=False)
	ax.scatter(plotData['lon'],plotData['lat'],zorder=1, alpha= 0.2, c='r', s=1)
	'''
	color = iter(cm.rainbow(np.linspace(0, 1, 33)))
	centroids = pd.DataFrame( columns = ['longitude','latitude'])
	for i in range(33):
		plotData = stops_labels.loc[stops_labels['label'] == i]
		plotData = plotData.set_axis(['lon', 'lat', 'label'], axis=1, inplace=False)


		cluster_point_lon = plotData['lon'].mean()
		cluster_point_lat = plotData['lat'].mean()

		c = next(color)
		ax.scatter(cluster_point_lon,cluster_point_lat,zorder=1, alpha= 1, c=c, s=5)
		centroids.loc[len(centroids)] = [cluster_point_lon,cluster_point_lat]

	#ax.scatter(cluster_point[:,0],cluster_point[:,1], zorder=1, alpha= 0.2, c='r', s=5)

	print(centroids)
	'''
	longt = listings.iloc[0]['longitude']
	lati = listings.iloc[0]['latitude']
	print(distanceToNearestCluster(longt,lati,centroids))
	'''

	listings['distanceToCluster'] = listings.apply(lambda row: distanceToNearestCluster(row['longitude'],row['latitude'],centroids), axis = 1)
	convenient_airbnb = listings.loc[listings['distanceToCluster'] < 0.5]
	print(convenient_airbnb)

	ax.scatter(convenient_airbnb['longitude'],convenient_airbnb['latitude'],zorder=1, alpha= 0.2, c='black', s=1)

	ax.set_title('Area with high density of bus stops')
	ax.set_xlim(-123.5,-122)
	ax.set_ylim(49,49.5)
	ax.imshow(map_van, zorder=0, extent = Boundary, aspect= 'equal')
	plt.show()

if __name__ == '__main__':
    main()

