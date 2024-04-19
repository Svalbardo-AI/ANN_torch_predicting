import numpy as np
import math

import pandas as pd
import os
import torch

root_path = os.getcwd()


def standardize(x, _max, _min) -> pd.DataFrame:
    return (x - _min) / (_max - _min)


def data_init() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    dataset = pd.read_csv(root_path + '\\' + 'std_000001.SZ.csv', index_col=False)
    cache_1 = dataset.iloc[:-10, :-1]
    cache_2 = dataset.iloc[:-10, -1:]
    cache_3 = dataset.iloc[-10:, :-1]
    cache_4 = dataset.iloc[-10:, -1:]
    return cache_1, cache_2, cache_3, cache_4


def cal_ssr(x: torch.tensor, y: torch.tensor) -> torch.float:
    return torch.sum((x - y) ** 2)
