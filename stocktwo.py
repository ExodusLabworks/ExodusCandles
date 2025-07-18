import yfinance as yf

def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def days_in_month(year, month):
    if month == 2:
        return 29 if is_leap_year(year) else 28
    return 31 if month in [1,3,5,7,8,10,12] else 30

def subtract_days(end_date, offset_days):
    year, month, day = map(int, end_date.strip().split('-'))

    while offset_days > 0:
        if offset_days >= day:
            offset_days -= day
            month -= 1
            if month == 0:
                month = 12
                year -= 1
            day = days_in_month(year, month)
        else:
            day -= offset_days
            offset_days = 0

    return f"{year:04d}-{month:02d}-{day:02d}"

def get_stock_data(ticker, end_date, offset_days):
    start_date = subtract_days(end_date, offset_days)

    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

        if stock_data.empty:
            print("No data found. Please check the ticker or date range.")
            return None

        # Handle both regular and MultiIndex column structures
        if ("Open", ticker) in stock_data.columns:
            open_col = ("Open", ticker)
            close_col = ("Close", ticker)
            high_col = ("High", ticker)
            low_col = ("Low", ticker)
            vol_col = ("Volume", ticker)
        else:
            open_col = "Open"
            close_col = "Close"
            high_col = "High"
            low_col = "Low"
            vol_col = "Volume"

        opens = stock_data[open_col].astype(float).tolist()
        closes = stock_data[close_col].astype(float).tolist()
        highs = stock_data[high_col].astype(float).tolist()
        lows = stock_data[low_col].astype(float).tolist()
        volumes = stock_data[vol_col].astype(int).tolist()
        dates = [d.strftime('%Y-%m-%d') for d in stock_data.index]

        return dates, opens, closes, highs, lows, volumes

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    ticker = input("Enter NYSE ticker symbol (e.g., AAPL): ").strip().upper()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()
    days_prior = int(input("Enter number of days PRIOR to end date (30, 60, 90, 120, 180): ").strip())

    result = get_stock_data(ticker, end_date, days_prior)

    if result:
        dates, opens, closes, highs, lows, volumes = result
        print(opens[20])
    else:
        print("Failed to retrieve stock data.")

if __name__ == "__main__":
    main()
