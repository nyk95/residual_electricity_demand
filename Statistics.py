# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import t
# K means
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler









# Cluster columns in dataframe using k-means clustering and return labels to model object indicating which clusters each column belongs to
def cluster(m, df, name):
    
    features = np.array(df).T
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    # Initialize kmeans object
    kmeans = KMeans(
        init="random",
        n_clusters=m.n_clusters,
        n_init=10,
        max_iter=300,
        random_state=42
    )
    # Determine clusters
    kmeans.fit(scaled_features)
    # Labels indicating which data points belong together in clusters
    labels = kmeans.labels_[:]
    df = pd.DataFrame(labels, columns=[name], index=m.geo_points.index)
    m.geo_points = m.geo_points.join(df)


# Perform population weighted mean of columns given the cluster labels are given in model object
def pop_weighted_mean(m, df, name):
    # takes the temprature matrix, and finds the column indexes for temprature 
    # time series which belong in the same cluster. All this to later take pop weighted mean
    features = np.array(df)    
    n = m.n_clusters
    # Array of temperature matrix column indices
    col = np.arange(np.shape(features)[1])
    # Put cluster labels and columns together
    a = np.vstack((m.geo_points[name].values, col)).T
    # sort it based on cluster label (0,1,2,...)
    a = a[a[:, 0].argsort()]
    # Split into arrays based on cluster labels : arrays of temperature matrix column index
    cluster_idx = np.split(a[:,1], np.unique(a[:, 0], return_index=True)[1][1:])
    
    cluster_names = [name + '_' + str(x) for x in np.unique(a[:, 0])]
    # Calculate total population weight of each cluster 
    cluster_weights = m.geo_points.groupby(name).sum()['Pop_weight'].to_dict()
    # Calculate the weight of each point within each cluster
    weights = np.divide(m.geo_points['Pop_weight'].values, [cluster_weights[p] for p in m.geo_points[name]])

    # Define new array with mean values of columns that belong together in clusters
    mean_features = np.zeros((np.shape(features)[0], n))
    for i in range(0, n):
        mean_features[:,i] = np.sum(features[:, cluster_idx[i]] * weights[cluster_idx[i]], axis=1)

    df = pd.DataFrame(mean_features, columns=cluster_names, index=df.index)

    return df


# Select cluster variables from data frame based on name string and scale them according to popolation weight of each cluster
# used to give country level (total) pop weighted mean from each cluster temp time series 
def pop_weighted_mean_total(m, df, string):
    # Find columns with temperature clusters
    idx = df.columns[df.columns.str.startswith(string)]
    df_var = np.array(df[idx])
    # Population weights for each cluster
    weights = m.geo_points.groupby('Temp_cluster').sum()['Pop_weight'].values
    # Population weighted mean across clusters
    mean = np.sum(df_var * weights, axis=1)
    df = pd.DataFrame({'Pop weight mean': mean}, index=df.index)

    return df



# Fit data to t distribution and return parameters
def fit_t(x):
    t_params = t.fit(x)
    return t_params

# Sample random variables from a t distribution
def sample_t(t_params, s):
    r = t.rvs(df=t_params[0], loc=t_params[1], scale=t_params[2], size=s)
    return r