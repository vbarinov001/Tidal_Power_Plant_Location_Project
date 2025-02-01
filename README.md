# Tidal_Power_Plant_Location_Project

## 📚 About Data

The dataset includes U.S. tidal stations with mean sea level (MSL) trends and U.S. power plant locations. The script processes and cleans the data, calculates the nearest power plant distance for each station using the Haversine formula, and ranks stations based on a combined score of MSL trends and proximity to power plants. A raster depth map is used to filter and visualize top-ranked stations within a specified depth range, aiding in identifying optimal tidal power locations.

## 💡 Highlights

- Sea Level Trends & Tidal Power: Analyzing U.S. tidal stations and power plants to identify optimal tidal energy locations.
- Geospatial Analysis: Utilizing GEBCO bathymetric data to assess proximity and feasibility.
- Data Processing & Scoring: Cleaning datasets and implementing a scoring system for site selection.

## ✏️ Data Wrangling

- Data Cleaning: Removed inconsistencies and missing values from tidal station and power plant datasets.
- Data Integration: Merged NOAA sea level trends with tidal power plant locations for geospatial analysis.
- Feature Engineering: Calculated proximity scores and site feasibility metrics using GEBCO bathymetric data.

## 📊 Visualization

This code generates a depth visualization of a geographic region using raster data, overlaying key tidal station locations and their nearest power plants while annotating depth values and station names for better analysis.

![Figure_1](https://github.com/user-attachments/assets/5ffcfbce-e2a0-48d9-b34c-8f2918ee480f)
![Figure_2_NW_REGION1](https://github.com/user-attachments/assets/83f75555-f68e-4d50-8b1d-6b068e9e9668)
![Figure_3_NW_REGION2](https://github.com/user-attachments/assets/641bb38b-0f1f-4a9b-9fb0-aa671d00165e)
![Figure_4_W_REGION](https://github.com/user-attachments/assets/9390a25c-f33a-429e-b0fe-829f6213e372)
![Figure_5_ISLAND_REGION](https://github.com/user-attachments/assets/3945167f-30cb-4f78-82a9-4710a2f51452)

[Location Analysis].(https://github.com/vbarinov001/Tidal_Power_Plant_Location_Project/blob/83c0faf8af7a02ebf87e4d66f437d9b363a8013f/TPP_DATA_ANALYSIS%20DRAFT.txt).
