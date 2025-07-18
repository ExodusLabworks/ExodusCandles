import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def get_ohlcv_csv(ticker):
    # Set output filename
    filename = f"{ticker}_ohlcv.csv"
    
    # Download OHLCV data for past 60 days
    end_date = datetime.today()
    start_date = end_date - timedelta(days=60)
    
    print(f" Downloading {ticker} data from {start_date.date()} to {end_date.date()}")
    
    try:
        stock = yf.Ticker(ticker)
        ohlcv = stock.history(start=start_date, end=end_date)
        
        if ohlcv.empty:
            print("!! No data found for that ticker.")
            return
        
        # Remove timezone information from the index
        ohlcv.index = ohlcv.index.tz_localize(None)
        
        ohlcv_filtered = ohlcv[["Open", "High", "Low", "Close", "Volume"]]
        ohlcv_filtered.index.name = "Date"  # Keep the index column named "Date"
        
        # Save to CSV
        ohlcv_filtered.to_csv(filename)
        print(f" Saved to: {filename}")
        
    except Exception as e:
        print(f" Error downloading data: {e}")

if __name__ == "__main__":
    ticker = input("Enter stock ticker (e.g., AAPL, MSFT, SPY): ").upper().strip()
    if ticker.isalpha():
        get_ohlcv_csv(ticker)
    else:
        print(" Invalid input. Please enter a valid stock ticker.")