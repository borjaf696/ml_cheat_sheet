import osmnx as ox
import numpy as np
import haversine as hs
import networkx as nx
from sklearn.metrics.pairwise import haversine_distances
from tester_interface import Tester

class GeographicalTester(Tester):
    @abstractmethod
    def do_evaluation(self, model, df):
        pass

    @abstractmethod
    def do_models_comparison(self, models, df_test, **kwargs):
        pass

class GraphGeographicalTester(GeographicalTester):
    # The ground truth should be in a different class, since we might want to change it.
    def __calculate_ground_truth(self, df, show = True):
        centroid_lat, centroid_lon = df.lat.mean(), df.lon.mean()
        if show:
            print('Centroid located at: ', (centroid_lat, centroid_lon))
            print('Radius length: ', d_objective)
        d = list(haversine_distances(df[['lon_rad','lat_rad']], df[['lon_rad','lat_rad']])*R*1000)[0]
        d_objective = float(np.max(d))/2
        G_drive = ox.graph_from_point((centroid_lat, centroid_lon), dist = d_objective, simplify=True,network_type='drive')
        G_walk = ox.graph_from_point((centroid_lat, centroid_lon), dist = d_objective, simplify=True,network_type='walk')
        G_walking_simplified = ox.graph_from_point((centroid_lat, centroid_lon), dist = d_objective, simplify=True,network_type='walk'\
            , custom_filter='["highway"~"motorway"]')
        return {'drive':G_drive, 'walk':G_walk, 'graph_walk_simplified':G_walking_simplified}

    def __init__(self, df):
        Tester.__init__(data = df, name = 'GraphGeographicalTester')
        self._ground_truth = self.__calculate_ground_truth(df)
        self._engine, self._custom_model_flag = 'OSM', 'CUSTOM'

    def _get_all_models(self, models):
            dict_graphs = dict()
            for key, val in self._ground_truth.items():
                dict_graphs[key] = (self._engine, val)
            for key,val in models.items():
                dict_graphs[key] = (self._custom_model_flag, val)
            return dict_graphs
        
    @abstractmethod
    def do_evaluation(self, model, df):
        print('Launching graphs evaluation method for Geographical Data')
        return df

    @abstractmethod
    def do_models_comparison(self, models, df_test, **kwargs):
        '''
        Models comparison fucntion from a dictionary of graphs and a dataframe. 
        It builds in a reproducible way the test dataframe.
        Params:
            models:
                - key: name of the dict
                - val: a tuple ({'OSM','CUSTOM'}, graph)
            df_test - dataframe for testing
        Return:
            dicts_result:
                - key: name of the dict (1:1 with dir_graphs + naive_metrics)
                - val: distances for the paths
            dicts_paths:
                - key: name of the dict (1:1 with dir_graphs + naive_metrics)
                - val: set of coordinates for each path
        '''
        def __node_list_to_path(G, node_list):
            """
            Given a list of nodes, return a list of lines that together
            follow the path
            defined by the list of nodes.
            Parameters
            ----------
            G : networkx multidigraph
            route : list
                the route as a list of nodes
            Returns
            -------
            lines : list of lines given as pairs ( (x_start, y_start), 
            (x_stop, y_stop) )
            """
            edge_nodes = list(zip(node_list[:-1], node_list[1:]))
            lines = []
            for u, v in edge_nodes:
                # if there are parallel edges, select the shortest in length
                data = [G.get_edge_data(u,v)][0]
                # if it has a geometry attribute
                if 'geometry' in data:
                    # add them to the list of lines to plot
                    xs, ys = data['geometry'].xy
                    lines.append(list(zip(xs, ys)))
                else:
                    x1 = G.nodes[u]['x']
                    y1 = G.nodes[u]['y']
                    x2 = G.nodes[v]['x']
                    y2 = G.nodes[v]['y']
                    line = [(x1, y1), (x2, y2)]
                    lines.append(line)
            return lines

        col = kwargs['col']
        dict_graphs = self._get_all_models(models)
        naive_metrics = ['euclidean_dist', 'circular_dist']
        results_keys = list(dict_graphs.keys()) + naive_metrics
        dicts_result = {key:[] for key in results_keys}
        dicts_paths = {key:[] for key in results_keys}
        start_point = (float(df_test.iloc[0]['lat']),float(df_test.iloc[0]['lon']))
        start_point_cluster = df_test.iloc[0][col]
        for i, _ in enumerate(df_test.iterrows()):
            if i == 0:
                continue
            destination = (float(df_test.iloc[i]['lat']),float(df_test.iloc[i]['lon']))
            destination_point_cluster = df_test.iloc[i][col]
            dicts_result['euclidean_dist'].append(ox.distance.euclidean_dist_vec\
                (start_point[0], start_point[1], destination[0], destination[1])*10**5)
            dicts_result['circular_dist'].append(ox.distance.great_circle_vec\
                (start_point[0], start_point[1], destination[0], destination[1]))
            for key, g_tmp in dict_graphs.items():
                try:
                    if g_tmp[0] == self._engine:
                        origin_id = ox.nearest_nodes(g_tmp[1], start_point[1], start_point[0])
                        destination_id = ox.nearest_nodes(g_tmp[1], destination[1], destination[0])
                    else:
                        origin_id, destination_id = start_point_cluster, destination_point_cluster
                    original_path = nx.shortest_path(g_tmp[1], origin_id, destination_id)
                    path, path_distance = __node_list_to_path(g_tmp[1], original_path), 0
                    edges = list(zip(original_path[:-1], original_path[1:]))
                    for edge in edges:
                        if g_tmp[0] == self._engine:
                            path_distance += g_tmp[1].get_edge_data(edge[0],edge[1])[0]['length']
                        elif g_tmp[0] == self._custom_model_flag:
                            path_distance += g_tmp[1].get_edge_data(edge[0],edge[1])['length']
                    dicts_result[key].append(path_distance)
                    dicts_paths[key].append(path)
                except Exception as e:
                    # logging.info('Graph: '+str(key)+' Exception: '+str(e))
                    dicts_result[key].append(-1)
                    dicts_paths[key].append([-1])
        return dicts_result, dicts_paths, df_test