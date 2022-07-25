import time
import sys, os
import pandas as pd
import geopy
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
import pysal
import contextily

import osmnx as ox
import numpy as np
import datetime

import haversine as hs
import networkx as nx

def _to_cartesian(loc_degrees):
    '''
    loc_degrees = (lat, lon)
    '''
    R = 6371*10**3
    loc = np.radians(loc_degrees[0]), np.radians(loc_degrees[1])
    return R*np.cos(loc[0])*np.cos(loc[1]), R*np.sin(loc[1])*np.cos(loc[0]), R*np.sin(loc[0])

def _to_cartesian_df(loc, col = ['lon','lat']):
    '''
    loc = lon, lat
    '''
    R = 6371
    return R*np.cos(loc[col[0]])*np.cos(loc[col[1]]), \
        R*np.sin(loc[col[1]])*np.cos(loc[col[0]]), R*np.sin(loc[col[0]])

def get_distance(x, method = 'haversine'):
    '''
    x - dataframe row with columns:
        * lat_cluster, lon_cluster (degrees)
        * next_lat_cluster, next_lon_cluster (degrees)
    Return:
        * Distance in meters
    '''
    if x.next_lat_cluster == -1:
        return 0.
    if method == 'haversine':
        return hs.haversine((float(x.lat_cluster), float(x.lon_cluster))\
                , (float(x.next_lat_cluster), float(x.next_lon_cluster)))*1e3
    elif method == 'manhattan':
        loc_1 = _to_cartesian((float(x.lat_cluster), float(x.lon_cluster)))
        loc_2 = _to_cartesian((float(x.next_lat_cluster), float(x.next_lon_cluster)))
        return np.abs(loc_1[0] - loc_2[0]) + \
            np.abs(loc_1[1] - loc_2[1]) + np.abs(loc_1[2] - loc_2[2])
    elif method == 'euclidean':
        loc_1 = _to_cartesian((float(x.lat_cluster), float(x.lon_cluster)))
        loc_2 = _to_cartesian((float(x.next_lat_cluster), float(x.next_lon_cluster)))
        return np.sqrt((loc_1[0] - loc_2[0])**2 + \
            (loc_1[1] - loc_2[1])**2 + (loc_1[2] - loc_2[2])**2)
    sys.exit()

def __distance_measure(v_1, v_2, coords = False, method = 'euclidean'):
    '''
    Multiple distances
    Multiple dimensional
    '''
    def __hav(d):
        return (1 - np.cos(d))/2
    R = 6371*10**3
    if method == 'haversine':
        assert coords
        hav = 2*R*np.arcsin(np.sqrt(__hav(np.radians(v_1[0] - v_2[0]))\
            + (1 - __hav(np.radians(v_2[0] - v_1[0])) \
                - __hav(np.radians(v_2[0] + v_1[0])))\
                    * __hav(np.radians(v_1[1] - v_2[1])))
            )
        print('Haversine custom: ',hav)
        return hs.haversine(v_1, v_2)*1e3
    if coords:
        v_1,v_2 = _to_cartesian(v_1), _to_cartesian(v_2)
        print(v_1, v_2)
    distance = 0.
    for i, _ in enumerate(v_1):
        if method == 'euclidean':
            distance += (v_1[i] - v_2[i])**2
        elif method == 'manhattan':
            distance += np.abs((v_1[i] - v_2[i]))
    return np.sqrt(distance) if method == 'euclidean' else distance

def _locate_closer_cluster(new_loc, clusters_location = None, \
    cols = ['lon','lat'], m = None, col = 'cluster_name', option = 'euclidean',boundary= 10**(-1),
    k = None):
    '''
    Locate the closest cluster
    option = {0 (classic, hand made), 1 (haversine)}
    '''
    assert clusters_location is not None
    assert m is not None
    loc_lon, loc_lat = float(new_loc[cols[0]]), float(new_loc[cols[1]])
    # Filter the matrix distance
    agg_tmp = pd.DataFrame()
    j = 0
    while agg_tmp.shape[0] == 0:
        j +=1
        boundary = 10**(-CURR_PREC + j)
        agg_tmp = clusters_location.loc[(clusters_location[cols[0]] <= (loc_lon + boundary)) & \
                (clusters_location[cols[0]]>= (loc_lon - boundary)) &\
                (clusters_location[cols[1]] <= (loc_lat + boundary)) &\
                (clusters_location[cols[1]] >= (loc_lat - boundary))]
    if option == 'naive':
        x_pos, y_pos, z_pos = _to_cartesian_df(new_loc[cols]*np.pi/180)
        matrix_distance = __distance_measure(m, [x_pos, y_pos, z_pos])
    # Euclidean
    elif option == 'euclidean':
        matrix_distance = np.array((agg_tmp[cols[0]] - new_loc[cols[0]])**2 + (agg_tmp[cols[1]] - new_loc[cols[1]])**2)
    # Manhattan
    elif option == 'mahnattan':
        matrix_distance = np.array((agg_tmp[cols[0]] - new_loc[cols[0]])) + np.array((agg_tmp[cols[1]] - new_loc[cols[1]]))
    if k is None:
        min_idx = np.argmin(matrix_distance, axis = 0)
    else:
        min_idx = np.argpartition(matrix_distance, kth = -(k + 1), axis = -1)
    return int(agg_tmp.iloc[min_idx][col])

def _get_distances(locs_1, locs_2, method = 'manhattan'):
    '''
    locs* = (lon, lat)
    '''
    R = 6371
    d_lat = locs_2[1] - locs_1[1]  
    d_lon = locs_2[0] - locs_1[0]
    if locs_1[0] > 2*np.pi:
        d_lat *= np.pi/180
        d_lon *= np.pi/180
    if method == 'euclidean':
        a = np.sin(d_lat/2)**2 + np.cos(locs_1[1]) * np.cos(locs_2[1])*np.sin(d_lon/2)**2
        c = 2*np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return np.round(R*c,2)
    elif method == 'manhattan':
        a_lat, a_lon = np.sin(d_lat/2)**2, np.sin(d_lon/2)**2
        c_lat, c_lon = 2*np.arctan2(np.sqrt(a_lat), np.sqrt(1 - a_lat)), \
            2*np.arctan2(np.sqrt(a_lon), np.sqrt(1 - a_lon))
        r_lat, r_lon = R*c_lat, R*c_lon
        return np.abs(r_lat) + np.abs(r_lon)

def save_graph(graph,file_name):
    from matplotlib import pylab
    #initialze Figure
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph,pos)
    nx.draw_networkx_edges(graph,pos)
    nx.draw_networkx_labels(graph,pos)

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)

    plt.savefig(file_name,bbox_inches="tight")
    pylab.close()
    del fig

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

def custom_cov(X):
    return np.matmul((X - np.mean(X, axis = 0)).T,((X - np.mean(X, axis = 0))))/X.shape[0]

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, labels = None,label=True, ax=None):
    ax = ax or plt.gca()
    covars = []
    if labels is not None:
        for cluster in np.unique(labels):
            idx = (labels == cluster)
        covars.append(custom_cov(X[idx]))
    else:
        covars.append(custom_cov(X))
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, covars, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)