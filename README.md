# Attention-based CNN-LSTM and XGBoost hybrid model for stock prediction

## Requirements

The code has been tested running under Python 3.7.4, with the following packages and their dependencies installed:
```
numpy==1.16.5
sklearn==0.21.3
statsmodels==0.10.1
pandas==0.25.1
tensorflow==2.1.0
keras==2.3.1
xgboost==1.5.0
```

The stock data used in this repository was downloaded from [TuShare](https://tushare.pro/). The stock data on [TuShare](https://tushare.pro/) are with public availability.

## Usage

Firstly, run `ARIMA.py` for pre-processing step by ARIMA model. Then, run the neural network or XGBoost models.
- Run `LSTM.py` for the single-layer LSTM, multi-layer LSTM, and bidirectional LSTM models.
- Run `XGBoost.py` for the XGBoost model.
- Run `Main.py` for our proposed Attention-based CNN-LSTM and XGBoost hybrid model.

## Citation
```
@article{shi2022attclx,
    author={Zhuangwei Shi and Yang Hu and Guangliang Mo and Jian Wu},
    title={Attention-based CNN-LSTM and XGBoost hybrid model for stock prediction},
    journal={arXiv preprint arXiv:2204.02623},
    year={2022},
}
```
