# Libraries
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely as sh
from scipy.spatial import cKDTree

# Custom libraries
import Setup as setup

# Find the nearest neighbour to gdB in gdA (gdA and gdB representing geo dataframes)
def ckd_nearest(gdA, gdB):
    nA = np.array(list(gdA.geometry.centroid.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.centroid.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ], 
        axis=1)

    return gdf

# Returns a dataframe with the geographical positions from the grid file located within an region in the polygon file
def points_in_geo(region):
    # Import region geometry
    gdf_polygons = gpd.read_file(setup.geo_polygon_file)
    gdf_region = gdf_polygons.loc[gdf_polygons['id'] == region]

    # Import sampling grid points and make a geo dataframe
    df_points = pd.read_excel(setup.grid_file)
    gdf_points = gpd.GeoDataFrame(df_points, crs='epsg:4326', geometry=gpd.points_from_xy(df_points.Longitude, df_points.Latitude))

    # Find points within geometry
    gdf_intersect_points = gpd.sjoin(gdf_points, gdf_region, how='inner')

    # Construct regular dataframe with desired information
    df = pd.DataFrame(gdf_intersect_points[['Name', 'Latitude', 'Longitude']])
    df = df.set_index('Name')

    # Import population data 
    pop = gpd.read_file(setup.population_file)
    pop = pop.to_crs('epsg:4326')

    # Select population point samples inside region
    pop_region = gpd.sjoin(pop, gdf_region, how='inner')

    # Associate all population samples to their nearest sampling points
    gdf_pop_points = ckd_nearest(pop_region.to_crs('epsg:7416'), gdf_intersect_points.to_crs('epsg:7416'))

    # Sum all population samples for each sampling point
    gdf_pop_points_sum = gdf_pop_points.drop(columns='geometry').groupby('Name').sum()

    # Put population weight into dataframe
    df['Pop_weight'] = gdf_pop_points_sum['TOT_P_2018'] / gdf_pop_points_sum['TOT_P_2018'].sum()
    
    return df


# Determine the number of clusters for a given region based on its latitudinal span
def n_clusters(region):
    gdf_polygons = gpd.read_file(setup.geo_polygon_file)
    gdf_region = gdf_polygons.loc[gdf_polygons['id'] == region]
    latitude_range = float((gdf_region.bounds['maxy'] - gdf_region.bounds['miny']).iloc[0])

    n = int(max(1, round(latitude_range*setup.clusters_per_degree_lat)))

    return n


def plot_clusters(m):
    # Import region geometry
    gdf_polygons = gpd.read_file(setup.geo_polygon_file)
    gdf_region = gdf_polygons.loc[gdf_polygons['id'] == m.region]

    # Import sampling grid points and make a geo dataframe
    gdf_points = gpd.GeoDataFrame(m.geo_points, crs='epsg:4326', geometry=gpd.points_from_xy(m.geo_points.Longitude, m.geo_points.Latitude))

    fig, ax = plt.subplots(1,1, figsize=(8,8))
    gdf_region.plot(ax=ax)
    gdf_points.plot(column='Temp_cluster', ax=ax, cmap='BuGn')
    ax.set_title(m.region + ' Clusters')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.grid(alpha=0.7)
    filename = 'Model validation/' + m.region + '/' + ' ' + m.region + ' Cluster map.png'
    plt.savefig(filename, dpi=300, transparent=False)
    #plt.show()
    plt.close()