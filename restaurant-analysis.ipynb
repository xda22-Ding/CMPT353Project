import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.weightstats as sw

food = pd.read_csv("restaurant-info.csv")
print(food)

food = food.dropna(how='any')
food = food.drop_duplicates()
print(food)

Chain = food[food.NewChain == True]
NotChain = food[food.NewChain == False]

print(Chain.name.count(), NotChain.name.count())
print(Chain.lat.mean(), NotChain.lat.mean())
print(Chain.lon.mean(), NotChain.lon.mean())
print(Chain.lat.std(), NotChain.lat.std())
print(Chain.lon.std(), NotChain.lon.std())

plt.figure()

print(Chain.lat.mean(), Chain.lat.std())
print(Chain.lat.min(), Chain.lat.max())
plt.subplot(2, 2, 1)
plt.boxplot(Chain.lat, notch=True, vert=False)
plt.title('Chain Restaurants Latitude')

print(Chain.lon.mean(), Chain.lon.std())
print(Chain.lon.min(), Chain.lon.max())
plt.subplot(2, 2, 2)
plt.boxplot(Chain.lon, notch=True, vert=False)
plt.title('Chain Restaurants Longitude')

print(NotChain.lat.mean(), NotChain.lat.std())
print(NotChain.lat.min(), NotChain.lat.max())
plt.subplot(2, 2, 3)
plt.boxplot(NotChain.lat, notch=True, vert=False)
plt.title('Non-Chain Restaurants Latitude')

print(NotChain.lon.mean(), NotChain.lon.std())
print(NotChain.lon.min(), NotChain.lon.max())
plt.subplot(2, 2, 4)
plt.boxplot(NotChain.lon, notch=True, vert=False)
plt.title('Non-Chain Restaurants Longitude')
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12, 6))
plt.plot(NotChain['lon'], NotChain['lat'], 'g.', alpha=0.3)
plt.title('Density for NonChain Restaurant')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

fig = plt.figure(figsize=(12, 6))
plt.plot(Chain['lon'], Chain['lat'], 'r.', alpha=0.3)
plt.title('Density for Chain Restaurant')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

fig = plt.figure(figsize=(12, 6))
plt.plot(NotChain['lon'], NotChain['lat'], 'g.', alpha=0.6)
plt.plot(Chain['lon'], Chain['lat'], 'r.', alpha=0.6)
plt.title('Density for NonChain and Chain Restaurant (put together)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(['Non-Chain', 'Chain'])
plt.show()

from sklearn.cluster import DBSCAN

plt.figure()
plt.subplot(2, 1, 1)
chain_labels = DBSCAN(eps=0.03, min_samples=3).fit_predict(Chain[['lon', 'lat']])
plt.scatter(Chain['lon'], Chain['lat'], c=chain_labels)
plt.title('Chain Restaurant Clustering')

plt.subplot(2, 1, 2)
chain_labels = DBSCAN(eps=0.03, min_samples=3).fit_predict(NotChain[['lon', 'lat']])
plt.scatter(NotChain['lon'], NotChain['lat'], c=chain_labels)
plt.title('Non-Chain Restaurant Clustering')
plt.tight_layout()
plt.show()

print(sw.ztest(Chain['lat'], NotChain['lat'], alternative="two-sided"))
print(sw.ztest(Chain['lon'], NotChain['lon'], alternative="two-sided"))


