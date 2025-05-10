# Jeffrey Wong | ECE-478 | PSet #5
# Time Series- Financial Data Acquisition

# Acquires S&P 500 stock prices for a given time frame and calculates mean and correlation to find heteroskadasticity

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime

def download_SP500_data(start_date, end_date):
    # Select subset of two stocks from S&P 500
    symbol_list = ["IBM", "AAPL", "^SPX"]

    # Setting auto-adjust 
    SP_500_data = yf.download(symbol_list, start = start_date, end = end_date, interval = "1d", auto_adjust = False)
    SP_adjusted_close = SP_500_data["Adj Close"]
    return SP_adjusted_close

def get_log_returns(start_date, end_date):
    print("Retrieving S&P 500 data")
    SP_adjusted_close = download_SP500_data(start_date, end_date)
    print(SP_adjusted_close)
    print("Converting to log returns")
    SP_adjusted_close_data = SP_adjusted_close.to_numpy(copy = True)
    SP_log_returns_data = np.log(np.divide(SP_adjusted_close_data[1:], SP_adjusted_close_data[:-1]))
    print(SP_log_returns_data)
    trading_days = SP_adjusted_close.index.tolist()
    SP_log_returns = pd.DataFrame(data = SP_log_returns_data, index = trading_days[:-1], columns= SP_adjusted_close.columns)
    return SP_log_returns

def main():
    start_date = "2021-12-31"
    end_date = "2024-01-01"
    print("Finding log returns of stock:")
    SP_log_returns = get_log_returns(start_date, end_date)
    SP_log_returns.to_csv("SP500_log_returns.csv")
    plt.figure()
    plt.plot(SP_log_returns)
    plt.title("Stock returns over time")
    plt.ylabel("Log return")
    plt.savefig("returns.png")

    plt.figure()
    plt.plot(np.power(SP_log_returns, 2))
    plt.title("Squared returns over time")
    plt.ylabel("Log return squared")
    plt.savefig("squared_returns.png")

    # Each stock is a column so take mean along column
    SP_centered_log_returns = SP_log_returns - np.mean(SP_log_returns, axis = 0)
    print(SP_centered_log_returns)
    stock1_returns = SP_centered_log_returns.to_numpy(copy = True)[:,0]
    stock2_returns = SP_centered_log_returns.to_numpy(copy = True)[:,1]
    SP500_returns = SP_centered_log_returns.to_numpy(copy = True)[:,2]
    stock1_square_returns = np.power(stock1_returns,2)
    stock2_square_returns = np.power(stock2_returns,2)
    SP500_square_returns = np.power(SP500_returns,2)
    corr_length = stock1_returns.size
    # np.correlate gives the cross-correlation as specified at the beginning of the homework,
    # get autocorrelation by using the same argument twice and normalizing
    stock1_return_corr = np.correlate(stock1_returns, stock1_returns, "full")/corr_length
    stock2_return_corr = np.correlate(stock2_returns, stock2_returns, "full")/corr_length
    SP500_return_corr = np.correlate(SP500_returns, SP500_returns, "full")/corr_length

    # Normalize correlations
    stock1_return_corr = stock1_return_corr / np.max(stock1_return_corr)
    stock2_return_corr = stock2_return_corr / np.max(stock2_return_corr)
    SP500_return_corr = SP500_return_corr / np.max(SP500_return_corr)

    stock1_square_return_corr = np.correlate(stock1_square_returns, stock1_square_returns, "full")/corr_length
    stock2_square_return_corr = np.correlate(stock2_square_returns, stock2_square_returns, "full")/corr_length
    SP500_square_return_corr = np.correlate(SP500_square_returns, SP500_square_returns, "full")/corr_length

    stock1_square_return_corr = stock1_square_return_corr / np.max(stock1_square_return_corr)
    stock2_square_return_corr = stock2_square_return_corr / np.max(stock2_square_return_corr)
    SP500_square_return_corr = SP500_square_return_corr / np.max(SP500_square_return_corr)

    plt.figure()
    plt.stem(np.arange(21),np.abs(stock1_return_corr[corr_length-1:corr_length+20]), markerfmt = 'blue')
    plt.stem(np.arange(21),np.abs(stock2_return_corr[corr_length-1:corr_length+20]), markerfmt = 'lime')
    plt.stem(np.arange(21),np.abs(SP500_return_corr[corr_length-1:corr_length+20]), markerfmt = 'orange')
    plt.title("Stock return autocorrelations by lag")
    plt.ylabel("gamma(k)/gamma(0)")
    plt.xlabel("Timelag (m)")
    plt.axhline(0.2, ls = "--", c = 'red')
    plt.savefig("r_autocorrelation.png")

    plt.figure()
    plt.stem(np.arange(21),np.abs(stock1_square_return_corr[corr_length-1:corr_length+20]), markerfmt = 'blue')
    plt.stem(np.arange(21),np.abs(stock2_square_return_corr[corr_length-1:corr_length+20]), markerfmt = 'lime')
    plt.stem(np.arange(21),np.abs(SP500_square_return_corr[corr_length-1:corr_length+20]), markerfmt = 'orange')
    plt.title("Stock squared return autocorrelations by lag")
    plt.ylabel("gamma(k)/gamma(0)")
    plt.xlabel("Timelag (m)")
    plt.axhline(0.2, ls = "--", c = 'red')
    plt.savefig("squared_r_autocorrelation.png")
    return 0

if __name__ == "__main__":
    main()