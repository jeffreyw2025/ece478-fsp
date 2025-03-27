# Jeffrey Wong | ECE-478 | PSet #2
# Portfolio Analysis- Data Acquisition

# Acquires S&P 500 stock prices for a given time frame and US T-bill data to calculate excess returns for stocks

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime

def download_SP500_data(start_date, end_date):
    # Select subset of stocks (5 tech + 5 health stocks) plus the S&P 500 index (^SPX)
    symbol_list = ["AAPL", "CSCO", "INTC", "QCOM", "IBM", "CI", "CVS", "DGX", "MRNA", "RMD", "^SPX"]

    # Setting auto-adjust 
    SP_500_data = yf.download(symbol_list, start = start_date, end = end_date, interval = "1d", auto_adjust = False)
    SP_adjusted_close = SP_500_data["Adj Close"]
    return SP_adjusted_close

# Taken from given code snippets, returns YEARLY interest rates (need to normalize later)
def download_Tbill_data(start_date, end_date):
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    fed_data= web.DataReader(['TB3SMFFM','FEDFUNDS'],'fred')
    fed_data['3MO T-BILL']=fed_data['TB3SMFFM']+fed_data['FEDFUNDS']
    fed_3mo= (fed_data['3MO T-BILL'].resample(rule='B').ffill().to_frame())/(100)
    return fed_3mo

def get_ex_returns(start_date, end_date):
    print("Retrieving S&P 500 data")
    SP_adjusted_close = download_SP500_data(start_date, end_date)
    print(SP_adjusted_close)
    print("Converting to percentage returns")
    SP_adjusted_close_data = SP_adjusted_close.to_numpy(copy = True)
    SP_returns_data = np.divide((SP_adjusted_close_data[1:] - SP_adjusted_close_data[:-1]), SP_adjusted_close_data[:-1])
    print(SP_returns_data)
    trading_days = SP_adjusted_close.index.tolist()
    print("Retrieving T-Bill data")
    fed_3mo = download_Tbill_data(start_date, end_date)
    fed_3mo_daily = fed_3mo / SP_adjusted_close.shape[0] # Divide by number of trading days in interest period
    fed_3mo_final = fed_3mo_daily.loc[trading_days[:-1]]
    print("Calculating excess returns")
    SP_ex_returns_data = SP_returns_data - fed_3mo_final.to_numpy(copy = True)
    SP_ex_returns = pd.DataFrame(data = SP_ex_returns_data, index = trading_days[:-1], columns= SP_adjusted_close.columns)
    print(SP_ex_returns)
    return SP_ex_returns

def main():
    start_date = "2022-12-31"
    end_date = "2024-01-01"
    print("Finding excess returns of stock:")
    SP_ex_returns = get_ex_returns(start_date, end_date)
    SP_ex_returns.to_csv("SP500_ex_returns.csv")
    return 0

if __name__ == "__main__":
    main()