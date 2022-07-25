from abc import ABC, abstractmethod
from typing import List
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

#TODO: Split in different files
class Context():

    def __init__(self, cluster_algo: Clustering, SEED = 6543210) -> None:
        self._cluster_algo = cluster_algo
        self._SEED = SEED

    @property
    def strategy(self) -> Clustering:
        return self._cluster_algo

    @strategy.setter
    def strategy(self, cluster_algo: Clustering) -> None:
        self._cluster_algo = cluster_algo

    def cluster(self, df, features) -> None:
        print('Context: calling to the clustering method')
        self._cluster_algo.do_cluster(df, features)

class Clustering(ABC):

    def __init__(self, data, features):
        self._data = data
        self._features = features

    @abstractmethod
    def do_cluster(self, df):
        pass