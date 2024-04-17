import pandas as pd
import numpy as np
import os


def cal_ma(x):
    x[['ma5', 'ma10', 'ma20', 'v_ma5', 'v_ma10', 'v_ma20', 'change_5', 'change_10', 'change_20']] = 0
    for i in range(len(x)):  # len(dataframe)
        x['ma5'] = x['close'].rolling(window=5).mean()
        x['ma10'] = x['close'].rolling(window=10).mean()
        x['ma20'] = x['close'].rolling(window=20).mean()
        x['v_ma5'] = x['vol'].rolling(window=5).mean()
        x['v_ma10'] = x['vol'].rolling(window=10).mean()
        x['v_ma20'] = x['vol'].rolling(window=20).mean()
        x['change_5'] = x['change'].rolling(window=5).mean()
        x['change_10'] = x['change'].rolling(window=10).mean()
        x['change_20'] = x['change'].rolling(window=20).mean()


def cal_bool(x):
    pass


def cal_price_level(x):
    pass


def standardize(x):
    pass


if __name__ == '__main__':
    root_path = os.getcwd()
    dataset = pd.read_csv(root_path + '\\' + '000001.SZ.csv', index_col=0)
    dataset = dataset.sort_values('trade_date', ascending=True)  # 倒序，方便使用rolling创建其他特征。
    cal_ma(dataset)
    cal_bool(dataset)
    cal_price_level(dataset)

    # print(dataset.head())
    save = 0
    if save == 1: dataset.to_csv(root_path + '\\' + 'ex_000001.SZ.csv')

    standardize(dataset)

    if save == 1: dataset.to_csv(root_path + '\\' + 'std_000001.SZ.csv')
