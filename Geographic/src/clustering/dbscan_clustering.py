from abc import abstractmethod
from Geographic.src.clustering.cluster_interface import Clustering
import cluster_interface

class DBscan_clustering(Clustering):
    @abstractmethod
    def do_cluster(self, df):
        return super().do_cluster(df)