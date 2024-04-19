import functions as f
from model import Model
from multiprocessing.pool import Pool


# def main(x_train, y_train, x_true, y_true):
#     model = Model(x_train, y_train)
#     model.fit()
#     model.predict(x_true, y_true)


if __name__ == '__main__':
    x_train, y_train, x_true, y_true = f.data_init()
    # print(x_train.head(), y_train.head(), x_true.head(), y_true.head())
    model = Model(x_train, y_train)
    model.fit()
    model.predict(x_true, y_true)
