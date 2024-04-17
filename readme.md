This is a experimental project for multi-layer ANN by torch.tensor, because complicated ANN is totally unable to avoid overfit on simple dataset.
Therefore I decide to found this project to test the performance of higher-level ANN.


This project contains three parts: stock data, data processing, functions and model.

1.<Stock data> comes from python lib <tushare>, consisting of date, price, volume, etc.

2.<Data processing> is necessary for the model by removing the negative influence of dimension. It also creates some 0-1 features for better performance of the model.

3.<Functions> is a toolbag made by myself to simplify the main codes. It contains some data processing methods and calculating functions.

4.<Model> is main codes of this project, which uses processed data to fit and predict.
