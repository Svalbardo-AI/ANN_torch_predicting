import pandas as pd
import numpy as np
import os

root_path = os.getcwd()
dataset = pd.read_csv(root_path+'\\'+'000001.SZ.csv',index_col=0)

