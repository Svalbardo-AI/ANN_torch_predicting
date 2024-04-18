import numpy as np
import math

import pandas as pd


def standardize(x, _max, _min) -> pd.DataFrame:
    return (x - _min) / (_max - _min)
