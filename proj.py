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
import seaborn
seaborn.set()



def distanceToNearestCluster(point_lon,point_lat,centroids):
	minDistance = 1000
	for i in range(len(centroids)):
		if(minDistance > distance( centroids.iloc[i]['latitude'], centroids.iloc[i]['longitude'], point_lat, point_lon)):
			minDistance = distance( centroids.iloc[i]['latitude'], centroids.iloc[i]['longitude'], point_lat, point_lon)
	return minDistance

# Thanks to : https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
def distance(lat1, lon1, lat2, lon2):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 12742 * asin(sqrt(a)) #2*R*asin...


def main():
	#import the amenity data in Vancouver
	stops= pd.read_json('transport_stop-vancouver.json',lines = True)
	#import the bus stop data from translink
	print(len(stops))
	#import the airbnb listings data in Vancouver
	listings = pd.read_csv('listings.csv')

	#remove the listings other than selected area
	stops = stops.loc[(stops['lon'] > -123.5) & (stops['lon'] < -122) & (stops['lat'] > 49) & (stops['lat'] < 49.5)]

	#https://stackoverflow.com/questions/32464280/converting-currency-with-to-numbers-in-python-pandas
	#convert the currency column to type float
	listings['price_float'] = listings['price'].replace('[\$,]', '', regex=True).astype(float)
	listings = listings.loc[(listings['longitude'] > -123.5) & (listings['longitude'] < -122) & \
	(listings['latitude'] > 49) & (listings['latitude'] < 49.5) & \
	(listings['bedrooms'] == 1) & (listings['price_float'] < 200)
	]

	# Use DBSCAN to produce several clusters with dense bus stops
	# set reasonable parameter to make sure 35 areas are calculated
	# The DBSCAN use the ball_tree algorithm and haversine matric to deal with latitude and longitude
	X = np.array(stops[['lon','lat']])
	model = DBSCAN(eps=20/6371, min_samples=15, algorithm='ball_tree', metric='haversine').fit(X)
	cluster_point = model.labels_

	stops_labels = pd.DataFrame(X)
	stops_labels['label'] = cluster_point


	#Use the OSM image as plot background
	map_van = plt.imread('map.png')
	Boundary = (-123.5,-122,49,49.5)
	fig = plt.figure(figsize = (12,10))
	ax1 = plt.subplot(2,2,1)
	ax1.imshow(map_van, zorder=0, extent = Boundary, aspect= 'equal')
	ax1.scatter(stops['lon'],stops['lat'], zorder=1, alpha= 0.2, c='b', s=1)
	ax1.set_title('Bus Stops in Vancouver')
	'''
	plotData = stops_labels.loc[stops_labels['label'] == 2]
	plotData = plotData.set_axis(['lon', 'lat', 'label'], axis=1, inplace=False)
	ax.scatter(plotData['lon'],plotData['lat'],zorder=1, alpha= 0.2, c='r', s=1)
	'''
	color = iter(cm.rainbow(np.linspace(0, 1, 35)))
	centroids = pd.DataFrame( columns = ['longitude','latitude'])
	for i in range(35):
		plotData = stops_labels.loc[stops_labels['label'] == i]
		plotData = plotData.set_axis(['lon', 'lat', 'label'], axis=1, inplace=False)

		c = next(color)
		c=c.reshape(1,-1)

		ax2 = plt.subplot(2,2,2)
		ax2.scatter(plotData['lon'],plotData['lat'],zorder=1, alpha= 0.8, c=c, s=5)
		ax2.imshow(map_van, zorder=0, extent = Boundary, aspect= 'equal')
		ax2.set_title('Bus Stop After DBSCAN Clustering')
		ax2.set_xlim(-123.5,-122)
		ax2.set_ylim(49,49.5)

		cluster_point_lon = plotData['lon'].mean()
		cluster_point_lat = plotData['lat'].mean()



		ax3 = plt.subplot(2,2,3)
		ax3.scatter(cluster_point_lon,cluster_point_lat,zorder=1, alpha= 0.5, c=c, s=3)
		ax3.imshow(map_van, zorder=0, extent = Boundary, aspect= 'equal')
		ax3.set_title('Centroids of high density area of bus stops')
		ax3.set_xlim(-123.5,-122)
		ax3.set_ylim(49,49.5)
		centroids.loc[len(centroids)] = [cluster_point_lon,cluster_point_lat]

	#ax.scatter(cluster_point[:,0],cluster_point[:,1], zorder=1, alpha= 0.2, c='r', s=5)
	'''
	longt = listings.iloc[0]['longitude']
	lati = listings.iloc[0]['latitude']
	print(distanceToNearestCluster(longt,lati,centroids))
	'''

	listings['distanceToCluster'] = listings.apply(lambda row: distanceToNearestCluster(row['longitude'],row['latitude'],centroids), axis = 1)
	convenient_airbnb = listings.loc[listings['distanceToCluster'] < 0.5]
	convenient_airbnb.to_csv("ideal_airbnb.csv")


	ax4 = plt.subplot(2,2,4)
	ax4.imshow(map_van, zorder=0, extent = Boundary, aspect= 'equal')
	ax4.scatter(convenient_airbnb['longitude'],convenient_airbnb['latitude'],zorder=1, alpha= 0.2, c='black', s=1)
	ax4.set_title('Ideal Airbnb Distribution')
	ax4.set_xlim(-123.5,-122)
	ax4.set_ylim(49,49.5)
	plt.show()

if __name__ == '__main__':
    main()

