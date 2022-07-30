from abc import ABC, abstractmethod
from typing import List
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

#TODO: Split in different files
class Context():

    def __init__(self, tester: Tester, SEED = 6543210) -> None:
        self._tester = tester
        self._SEED = SEED

    @property
    def strategy(self) -> Tester:
        return self._tester

    @strategy.setter
    def strategy(self, tester: Tester) -> None:
        self._tester = tester

    def evaluate(self, df) -> None:
        print("Context: calling to the evaluation method")
        self._tester.do_evaluation(df)

    def compare_models(self, models, df_test, custom_error_sample = None, **kwargs):
        if custom_error_sample is not None:
            df_test = df_test.sample(frac = custom_error_sample, random_state = self._SEED, **kwargs)
        self._tester.do_models_comparison(models, df_test = df_test)

class Tester(ABC):

    def __init__(self, data, name: str):
        self._tester_name = name
        self._data = data

    @abstractmethod
    def do_evaluation(self, df):
        pass

    @abstractmethod
    def do_models_comparison(self, models, df_test):
        pass