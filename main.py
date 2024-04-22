import functions as f
from model import Model

if __name__ == '__main__':
    x_train, y_train, x_true, y_true = f.data_init()
    best_model_ssr = []
    for i in range(10):
        print('model: ', i + 1)
        model = Model(x_train, y_train)
        model.fit()
        ssr = model.predict(x_true, y_true)
        best_model_ssr.append(ssr)
    print(min(best_model_ssr))
