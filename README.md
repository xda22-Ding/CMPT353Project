# CMPT353Project
Tour Vancouver with OSM data
## Description
This project aims to provide useful information and advice for tourists in Metro Vancouver. Three aspects are discussed in this project. 
For public transportation, we extract relevant information from OSM, and perform DBSCAN clustering to find dense areas and the airbnb housings close to them. For Restaurants, ...
## Getting Started
### Dependencies
* install following packages before running the program
```bash
pip install numpy 
pip install pandas
pip install matplotlib
pip install geopy
pip install geopandas
pip install -U scikit-learn
pip install statsmodels
pip install exif
```
### Executing program
In the hdfs, run osm-transport_stop.py. The input is the osm file in the SFU compute cluster and the output is transport_ stop
```bash
spark-submit osm-transport_stop.py /courses/datasets/openstreetmaps transport_stop
```
Run the just-vancouver.py to extract vancouver data.
```bash
spark-submit just-vancouver.py transport_stop transport_stop-vancouver
```
Run the proj.py which takes transport_stop-vancouver.json and listings.csv as input files. It will output the file ideal_airbnb.csv
```bash
python3 proj.py transport_stop-vancouver.json listings.csv
```
Run the scenery_extract.py which take amenities-vancouver.json.gz as the input file
```bash
spark-submit scenery_extract.py amenities-vancouver.json.gz output
```
Run the attraction_analyse.py then choose one image from \CMPT353Project-main\images and text message from location.txt after the program starts executing
```bash
python3 attraction_analyse.py
```
Run the restaurant- analysis.py which take amenities-vancouver.json.gz as input file. 
```bash
python3 analysis.py amenities-vancouver.json.gz
```

## Acknowledgements
The course material provided by Professor Baker.

https://www150.statcan.gc.ca/n1/daily-quotidien/200602/dq200602a-eng.htm

https://stackoverflow.com/questions/4971269/how-to-pick-a-new-color-for-each-plotted-line-within-a-figure-in-matplotlib

https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206

https://stackoverflow.com/questions/32464280/converting-currency-with-to-numbers-in-python-pandas

https://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size/

https://towardsdatascience.com/easy-steps-to-plot-geographic-data-on-a-map-python-11217859a2db

https://medium.com/spatial-data-science/how-to-extract-gps-coordinates-from-images-in-python-e66e542af354

https://towardsdatascience.com/geocode-with-python-161ec1e62b89

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html
