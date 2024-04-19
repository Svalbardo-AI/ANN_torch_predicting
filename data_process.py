import pandas as pd
import numpy as np
import os
import datetime
import functions as f


def cal_ma(x) -> None:
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


def cal_bool(x) -> None:
    x[['label_change', 'label_change_5']] = 0
    x.loc[x['change'] > 0, ['label_change']] = 1
    x.loc[x['change_5'] > 0, ['label_change_5']] = 1
    x['label_change'].astype(np.int8)
    x['label_change_5'].astype(np.int8)


def cal_price_level(x) -> None:
    cache = int(x.close.min())
    x['label_y'] = x['close'] - cache
    x.label_y = np.round(x.label_y, decimals=0)
    x['label_y_5'] = x['label_y'].shift(-5)


def cal_week_and_weekday(x) -> None:
    x[['weeks', 'week_day']] = 0
    for i in range(len(x)):
        cache = datetime.datetime.strptime(str(x.trade_date.iloc[i]), '%Y%m%d')
        x.loc[i, ['week_day']] = cache.isoweekday()

        first_day = datetime.datetime(cache.year, 1, 1)
        week_start = (first_day.weekday() + 6) % 7
        days = (cache - first_day).days + 1
        week = (days + week_start - 2) // 7 + 1
        x.loc[i, ['weeks']] = week
    x['weeks'].astype(np.int8)
    x['week_day'].astype(np.int8)


def standardize(x) -> pd.DataFrame:
    std_features = x.columns.tolist()[:-6]
    std_features.append('label_y_5')
    print(std_features)
    cache: pd.DataFrame = x[std_features]

    for i in std_features:
        _max = cache[i].max()
        _min = cache[i].min()
        cache.loc[:, i] = f.standardize(cache[i], _max, _min)
    output = x
    output.loc[:, std_features] = cache
    return output


if __name__ == '__main__':
    root_path = os.getcwd()

    dataset = pd.read_csv(root_path + '\\' + '000001.SZ.csv')
    dataset = dataset.sort_values('trade_date', ascending=True)  # 倒序，方便使用rolling创建其他特征。
    cal_ma(dataset)
    cal_bool(dataset)
    cal_week_and_weekday(dataset)
    cal_price_level(dataset)

    dataset = dataset.iloc[20:-5, 1:]  # 处理后数据应删去缺项数据以及trade_date
    dataset.to_csv(root_path + '\\' + 'ex_000001.SZ.csv', index=False, index_label=False)

    std_dataset = standardize(dataset)  # 在不改变数据分布的条件下，将特征全部映射到(0,1)以内，从而消除不同量纲造成的影响。
    std_dataset.to_csv(root_path + '\\' + 'std_000001.SZ.csv', index=False, index_label=False)

    corr = std_dataset.corr(numeric_only=True)  # 通过相关矩阵观察各个变量的相关性
    corr.to_csv(root_path + '\\' + 'correlation_matrix.csv')

