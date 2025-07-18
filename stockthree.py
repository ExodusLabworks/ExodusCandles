def get_latest_trading_day():
    eastern = pytz.timezone("US/Eastern")
    now_et = datetime.now(eastern)

    if now_et.weekday() >= 5 or now_et.time() < datetime.strptime("09:30", "%H:%M").time():
        end_dt = now_et - timedelta(days=1)
        while end_dt.weekday() >= 5:
            end_dt -= timedelta(days=1)
    else:
        end_dt = now_et

    return end_dt.strftime("%Y-%m-%d")

def get_stock_data(ticker, end_date, num_trading_days):
    try:
        # Convert end date to datetime
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Estimate buffer to cover enough *trading* days
        buffer_days = int(num_trading_days * 1.5)
        start_dt = end_dt - timedelta(days=buffer_days)
        start_date = start_dt.strftime("%Y-%m-%d")

        print(f"Requesting data for {ticker} from {start_date} to {end_date}")

        # Download data
        stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

        if stock_data.empty:
            print("No data found. Please check the ticker or date range.")
            return None

        # Debug: Print out the first few rows of the stock data
        # print("Downloaded data (first 5 rows):")
        # print(stock_data.head())

        # Detect if it's a MultiIndex (multiple tickers or unusual formatting)
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = [' '.join(col).strip() for col in stock_data.columns.values]
            opens = stock_data[f"Open {ticker}"].astype(float).tolist()
            closes = stock_data[f"Close {ticker}"].astype(float).tolist()
            highs = stock_data[f"High {ticker}"].astype(float).tolist()
            lows = stock_data[f"Low {ticker}"].astype(float).tolist()
            dates = stock_data.index.strftime('%Y-%m-%d').tolist()
           
        else:
            opens = stock_data["Open"].astype(float).tolist()
            closes = stock_data["Close"].astype(float).tolist()
            highs = stock_data["High"].astype(float).tolist()
            lows = stock_data["Low"].astype(float).tolist()
            dates = stock_data.index.strftime('%Y-%m-%d').tolist()


        # print("Columns:", stock_data.columns.tolist())
        # print("Number of rows:", len(stock_data))
        # print("Length of opens:", len(opens))
        # print("Length of highs:", len(highs))
   


        return dates, opens, closes, highs, lows

    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def bullOrBear(result,d,ticker):
    start_array = result[2]
   # print(len(start_array))
    total_array_size = int(len(start_array))
    #print(total_array_size)
    start_value = start_array[0]
    
    end_value = start_array[total_array_size-1]
    

    if start_value > end_value:
        print(f"\n{ticker} for {d} days is experiencing a BEAR market (-):\n")
    elif end_value > start_value:
        print(f"\n{ticker} for {d} days is experiencing a BULL market (+):\n")
    else:
        print(f"\n{ticker} for {d} days is experiencing volatilty and/or limited change:\n")

def RSI(closes,d):
    
    day_RSI = []
    for i in range(14,d):
        endDay = i
        startDay = i-14
        cGains = 0
        cLoses = 0
        for l in range(startDay,endDay):
            if closes[l] > closes[l-1]:
                cGains= cGains + closes[l]
            else:
                cLoses = cLoses + closes[l]
        RS_VALUE = cGains/cLoses
        day_RSI.append(100-(100/(1+RS_VALUE)))
    return day_RSI


def prediction(opens,closes,highs,lows,rsi):
    accuracy = 0
    indicators = 0

    highWicksRatios = []
    lowWicksRatios = []
    wickRatios = []
    medians = []
    

    for i in range(len(opens)):
        if opens[i]>closes[i]: # indicator wicks for red days
            highWicksRatios.append(highs[i]/opens[i])
            lowWicksRatios.append(closes[i]/lows[i])

        if closes[i]>opens[i]: #indicator wicks for green days
            highWicksRatios.append(highs[i]/closes[i])    
            lowWicksRatios.append(opens[i]/lows[i])

        medians.append(abs(opens[i]+closes[i])/2)
    for i in range(len(highWicksRatios)):
        wickRatios.append(highWicksRatios[i]/lowWicksRatios[i])
        # print(f"{highWicksRatios[i]} || {lowWicksRatios[i]} || {wickRatios[i]} || {closes[i] - opens[i]}")

   

def main():
    print("\nWelcome to Exodus Analytics Data Analyzer...")
    print("Please choose from the following menu.")
    print("Current Version: v1.1.6\n")

    ticker = input("Enter NYSE ticker symbol (e.g., AAPL): ").upper()

    print("\nChoose how far back to retrieve trading data:")
    print("Enter [1] 30d prior ")
    print("Enter [2] 60d prior ")
    print("Enter [3] 90d prior ")
    print("Enter [4] 120d prior ")
    print("Enter [5] 180d prior ")
    userMenuChoice = int(input("Enter your choice: "))

    while userMenuChoice > 5 or userMenuChoice < 1:
        userMenuChoice = int(input("Invalid selection. Enter a number from 1 to 5: "))

    num_days = {1: 30, 2: 60, 3: 90, 4: 120, 5: 180}[userMenuChoice]

    # Get today's date as string
    today = get_latest_trading_day()


    result = get_stock_data(ticker, today, num_days)


    if result:
        dates, opens, closes, highs, lows = result
        displayVars = input(f"Do you want to display stock data for {ticker} over {num_days} days? [Y/N]: ")
        if displayVars.lower() == 'y':
            print(f"\n{len(dates)} trading days of data for {ticker}:\n")
            for i in range(len(dates)):
                print(f"{dates[i]} | Open: {opens[i]:.2f} | High: {highs[i]:.2f} | Low: {lows[i]:.2f} | Close: {closes[i]:.2f}")
     
        # Call your other functions here
        bullOrBear(result, num_days, ticker)
        rsiVals = RSI(closes, num_days)
       
        prediction(opens, closes, highs,lows,rsiVals)

    else:
        print("Failed to retrieve stock data.")


if __name__ == '__main__':
    import pandas as pd  # Make sure this is available
    import yfinance as yf
    import pytz
    from datetime import datetime, timedelta
    main()
       