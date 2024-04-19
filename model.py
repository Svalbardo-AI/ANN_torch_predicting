import torch
import numpy as np
import functions as f

class Model:
    def __init__(self, x_train, y_train, nodes_1=512, nodes_2=64, nodes_3=8):
        self.x = torch.tensor(np.array(x_train), dtype=torch.float)
        self.y = torch.tensor(np.array(y_train), dtype=torch.float)
        self.row_x, self.line_x = x_train.shape
        self.layer_1 = torch.randn((self.line_x, nodes_1), dtype=torch.float, requires_grad=True)
        self.b_1 = torch.randn(nodes_1, dtype=torch.float, requires_grad=True)

        self.layer_2 = torch.randn((nodes_1, nodes_2), dtype=torch.float, requires_grad=True)
        self.b_2 = torch.randn(nodes_2, dtype=torch.float, requires_grad=True)

        self.layer_3 = torch.randn((nodes_2, nodes_3), dtype=torch.float, requires_grad=True)
        self.b_3 = torch.randn(nodes_3, dtype=torch.float, requires_grad=True)

        self.output = torch.randn((nodes_3, 1), dtype=torch.float, requires_grad=True)
        self.b_y = torch.randn(1, dtype=torch.float, requires_grad=True)

        self.losses = []

    def fit(self, alpha=0.01, steps=1000) -> None:
        x = self.x
        y = self.y
        for i in range(steps):
            hidden_1 = torch.sigmoid(torch.matmul(x, self.layer_1) + self.b_1)
            hidden_2 = torch.sigmoid(torch.matmul(hidden_1, self.layer_2) + self.b_2)
            hidden_3 = torch.sigmoid(torch.matmul(hidden_2, self.layer_3) + self.b_3)
            y_predict = torch.sigmoid(torch.matmul(hidden_3, self.output) + self.b_y)

            loss = f.cal_ssr(y,y_predict)
            self.losses.append(loss.data.numpy())

            loss.backward()

            self.layer_1.data -= alpha * self.layer_1.grad
            self.b_1.data -= alpha * self.b_1.grad
            self.layer_2.data -= alpha * self.layer_2.grad
            self.b_2.data -= alpha * self.b_2.grad
            self.layer_3.data -= alpha * self.layer_3.grad
            self.b_3.data -= alpha * self.b_3.grad
            self.output.data -= alpha * self.output.grad
            self.b_y.data -= alpha * self.b_y.grad

            self.layer_1.grad.zero_()
            self.b_1.grad.zero_()
            self.layer_2.grad.zero_()
            self.b_2.grad.zero_()
            self.layer_3.grad.zero_()
            self.b_3.grad.zero_()
            self.output.grad.zero_()
            self.b_y.grad.zero_()

    def predict(self, x_true, y_true) -> None:
        x = torch.tensor(np.array(x_true), dtype=torch.float)
        hidden_1 = torch.sigmoid(torch.matmul(x, self.layer_1) + self.b_1)
        hidden_2 = torch.sigmoid(torch.matmul(hidden_1, self.layer_2) + self.b_2)
        hidden_3 = torch.sigmoid(torch.matmul(hidden_2, self.layer_3) + self.b_3)
        y_predict = torch.sigmoid(torch.matmul(hidden_3, self.output) + self.b_y)
        cache = torch.tensor(np.array(y_true))
        print('SSR: ', f.cal_ssr(cache, y_predict))
        print('y_true: ', y_true)
        print('y_predict', y_predict)


if __name__ == '__main__':
    pass
