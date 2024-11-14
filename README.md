# Logistic-route-optimisation
import requests

# MapMyIndia API Key
API_KEY = 'aa1b4b483090eb63a8eafab98a5530bc'

# Example eLoc for Durgapur (replace this with actual eLoc values)
eloc_code = "uoczs2"  # Replace with actual eLoc codes

# eLoc API URL
url = f"https://atlas.mapmyindia.com/api/places/eloc?eloc={eloc_code}"

# Make the API request
headers = {'Authorization': f'Bearer {API_KEY}'}
response = requests.get(url, headers=headers)

# Parse the response
location_data = response.json()

# Extract latitude and longitude
latitude = location_data.get('latitude')
longitude = location_data.get('longitude')
place_name = location_data.get('name')

# Display the location details
print(f"Place: {place_name}, Latitude: {latitude}, Longitude: {longitude}")
# Convert to DataFrame
import pandas as pd

df = pd.DataFrame(location_data, columns=['Place', 'Latitude', 'Longitude'])
print(df.head())
# Extract latitude and longitude as features
X = df[['Latitude', 'Longitude']].values
import sklearn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Apply KMeans with 3 clusters (you can change the number of clusters)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Add cluster labels to DataFrame
df['Cluster'] = kmeans.labels_

# Plot the clusters
plt.scatter(X[:, 1], X[:, 0], c=kmeans.labels_, cmap='viridis')
plt.title('K-Means Clustering of eLoc Places in Durgapur')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Scale the coordinates (optional)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.1, min_samples=2).fit(X_scaled)

# Add cluster labels to DataFrame
df['Cluster'] = dbscan.labels_

# Plot the clusters
plt.scatter(X[:, 1], X[:, 0], c=dbscan.labels_, cmap='plasma')
plt.title('DBSCAN Clustering of eLoc Places in Durgapur')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

import folium

# Create a map centered at Durgapur
map_center = [23.5204, 87.3119]
m = folium.Map(location=map_center, zoom_start=12)

# Add markers to the map, color-coded by cluster
for _, row in df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=row['Place'],
        icon=folium.Icon(color='blue' if row['Cluster'] == 0 else 'green' if row['Cluster'] == 1 else 'red')
    ).add_to(m)

# Save the map to an HTML file
m.save("durgapur_eloc_clusters_map.html")
