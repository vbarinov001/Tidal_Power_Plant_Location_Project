import rasterio
from pyproj import Transformer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# Path to USStationsLinearSeaLevelTrends.csv + Cleaning
stations_data_path = r"C:\Users\vbari\Desktop\Coding\Tidal Power Project\USStationsLinearSeaLevelTrends.csv"
stations_df = pd.read_csv(stations_data_path)
stations_df.columns = stations_df.columns.str.strip()

# Path to (US) Power_Plants.csv + Cleaning
plants_data_path = r"C:\Users\vbari\Downloads\Power_Plants.csv"
plants_df = pd.read_csv(plants_data_path)
plants_df.columns = plants_df.columns.str.strip()
plants_df['Longitude'] = plants_df['Longitude'].round(4)
plants_df['Latitude'] = plants_df['Latitude'].round(4)

# Path to your TIF file
map_path = r"C:\Users\vbari\Desktop\Coding\Tidal Power Project\GEBCO_07_Dec_2024_d3e6eee084c3\GEBCO_07_Dec_2024_d3e6eee084c3\gebco_2024_n80.0_s10.0_w-175.0_e-115.0.tif"

def plot_MASKstationnames_with_points(lat_lon_list, min_depth, max_depth, station_names, closest_stations):
    """
    Plots the depth raster and highlights specific locations given by latitude and longitude.

    Parameters:
        lat_lon_list (list of tuples): List of (Latitude, Longitude) pairs to mark on the map
    """
    with rasterio.open(map_path) as src:
        # Read the raster data
        depth_data = src.read(1)

        # Create a mask for the specified depth range
        masked_data = np.where((depth_data >= max_depth) & (depth_data <= min_depth), depth_data, np.nan)
        
        # Transform lat/lon (WGS84) to the raster's CRS (projected coordinates)
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        
        # Transform lat / lon to project coordinates and raster indices
        points = []
        for (lat, lon), name in zip(lat_lon_list, station_names):
            x, y = transformer.transform(lon, lat)  # Transform to projected coordinates
            col, row = ~src.transform * (x, y) 
            row, col = int(row), int(col)
        
            # Ensure point is within bounds
            if 0 <= row < src.height and 0 <= col <= src.width:
                depth = depth_data[row,col]
                points.append((lon, lat, depth, name))
            else:
                print(f"Warning: Point ({lat}, {lon}) is outside raster bounds.")
        
        # Transform closest stations for plotting
        closest_points = []
        for lat, lon in closest_stations:
            x, y = transformer.transform(lon, lat)
            col, row = ~src.transform * (x, y)
            row, col = int(row), int(col)
            if 0 <= row < src.height and 0 <= col <= src.width:
                closest_points.append((lon, lat))

        # Plot the masked raster
        fig, ax = plt.subplots(figsize=(10, 8))
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
        # Plot the masked data
        im = ax.imshow(masked_data, cmap='viridis', origin='upper', extent=extent)
        
        # Add a colorbar for depth
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Depth (meters)', fontsize=12)
        
        # Plot main points:
        for lon, lat, depth, name in points:
            ax.scatter(lon, lat, color='red', s=10)
            ax.text(lon, lat, f"{name}\n{depth:.1f} m", color='red', fontsize=6, ha='left', va='bottom')
        
        # Plot closest stations
        for lon, lat in closest_points:
            ax.scatter(lon, lat, color='lightblue', s=5, marker='x')

        # Update axis labels to show latitude and longitude
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        print("Generating plot")

        plt.show()

# Finding Nearest Power Plant
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the Earth using the Haversine formula.
    """
    R = 6371  # Radius of Earth in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c  # Distance in kilometers

# Calculate the nearest power plant distance for each station
def calculate_nearest_power_plant_distance(stations_df, plants_df):
    distances = []
    for _, station in stations_df.iterrows():
        station_lat, station_lon = station['Latitude'], station['Longitude']
        min_distance = plants_df.apply(
            lambda row: haversine_distance(station_lat, station_lon, row['Latitude'], row['Longitude']),
            axis=1
        ).min()
        distances.append(min_distance)
    stations_df['Nearest_Power_Plant_Distance_km'] = distances
    return stations_df

stations_df = calculate_nearest_power_plant_distance(stations_df, plants_df)

# Combine MSL Trends and distance to power plants into a scoring mechanism
def calculate_combined_score(stations_df):
    """
    Calculate a combined score for each station using MSL Trends and distance to the nearest power plant.
    Higher MSL Trends and shorter distances yield better scores.
    """
    # Normalize MSL Trends and distances
    max_trend = stations_df['MSL Trends (mm/yr)'].max()
    min_trend = stations_df['MSL Trends (mm/yr)'].min()
    max_distance = stations_df['Nearest_Power_Plant_Distance_km'].max()
    min_distance = stations_df['Nearest_Power_Plant_Distance_km'].min()

    stations_df['Normalized_Trend'] = (stations_df['MSL Trends (mm/yr)'] - min_trend) / (max_trend - min_trend)
    stations_df['Normalized_Distance'] = 1 - (
        (stations_df['Nearest_Power_Plant_Distance_km'] - min_distance) / (max_distance - min_distance)
    )

    # Calculate combined score (e.g., 70% weight for trend, 30% weight for distance)
    stations_df['Combined_Score'] = 0.7 * stations_df['Normalized_Trend'] + 0.3 * stations_df['Normalized_Distance']
    return stations_df

stations_df = calculate_combined_score(stations_df)


with rasterio.open(map_path) as src:
    # Filter stations within raster bounds
    stations_within_bounds = stations_df[
        (stations_df['Longitude'] >= src.bounds.left) &
        (stations_df['Longitude'] <= src.bounds.right) &
        (stations_df['Latitude'] >= src.bounds.bottom) &
        (stations_df['Latitude'] <= src.bounds.top)
    ]

    # Sort stations by MSL Trend in descending order (highest MSL Trend first)
    sorted_stations = stations_within_bounds.sort_values(by='MSL Trends (mm/yr)', ascending=False)

    # Select the top 25 stations
    top_25_stations = sorted_stations.head(25)

    # Create the lat_lon_list containing only the latitude and longitude of the top 25 stations
    lat_lon_list = list(zip(top_25_stations['Latitude'], top_25_stations['Longitude']))
    station_names = top_25_stations['Station Name'].tolist()

    print("Top 25 Stations Sorted by Combined Score:")
    print(top_25_stations[['Station Name', 'Latitude', 'Longitude', 'MSL Trends (mm/yr)', 'Nearest_Power_Plant_Distance_km', 'Combined_Score']])

    # Find the closest station for each of the top 25 stations
    closest_stations = []
    for _, station in top_25_stations.iterrows():
        station_lat, station_lon = station['Latitude'], station['Longitude']
        distances = stations_within_bounds.apply(
            lambda row: haversine_distance(station_lat, station_lon, row['Latitude'], row['Longitude']),
            axis=1
        )
        # Exclude the station itself (distance = 0) and find the closest one
        closest_station = stations_within_bounds.loc[distances[distances > 0].idxmin()]
        closest_stations.append((closest_station['Latitude'], closest_station['Longitude']))

    # Example usage
    plot_MASKstationnames_with_points(lat_lon_list, 0, -150, station_names, closest_stations)
    #Optimal tidal power station depth is -5 to -50 meters
