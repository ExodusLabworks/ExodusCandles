import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import time

def get_stock_data(ticker, days=60):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days * 2)
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty or len(df) < days:
        return None
    return df.tail(days).copy()

def build_dataset(df):
    opens = df["Open"].values
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    volumes = df["Volume"].values

    data = []
    for i in range(3, len(closes) - 1):
        open_price = opens[i]
        close_price = closes[i]
        high = highs[i]
        low = lows[i]
        volume = volumes[i]

        median = (open_price + close_price) / 2
        fib = sum(closes[i - 3:i]) / 3
        sma = sum(closes[i - 3:i]) / 3
        ema = closes[i] * 0.5 + closes[i - 1] * 0.3 + closes[i - 2] * 0.2

        if open_price >= close_price:
            alpha = high
            beta = low
            sigma = alpha - open_price
            phi = close_price - beta
        else:
            alpha = low
            beta = high
            sigma = beta - close_price
            phi = open_price - alpha

        signal_algo = 1 if phi > sigma else 0  # ← Your algorithm's directional logic
        next_close = closes[i + 1]
        pct_change = ((next_close - close_price) / close_price) * 100
        direction = 1 if pct_change > 0 else 0

        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume,
            'fib': fib,
            'sma': sma,
            'ema': ema,
            'sigma': sigma,
            'phi': phi,
            'median': median,
            'signal_algo': signal_algo,
            'pct_change': pct_change,
            'direction': direction
        })

    return pd.DataFrame(data)

def train_models(df):
    features = ['open', 'high', 'low', 'close', 'volume', 'fib', 'sma', 'ema', 'sigma', 'phi', 'signal_algo']
    X = df[features]
    y_dir = df['direction']
    y_pct = df['pct_change']

    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y_dir, test_size=0.2, random_state=42)
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_pct, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    reg = RandomForestRegressor(n_estimators=100, random_state=42)

    clf.fit(X_train_d, y_train_d)
    reg.fit(X_train_p, y_train_p)

    acc = accuracy_score(y_test_d, clf.predict(X_test_d))
    err = mean_absolute_error(y_test_p, reg.predict(X_test_p))

    return clf, reg, acc, err, X.iloc[[-1]]

def predict_next(clf, reg, latest_row):
    X_latest = latest_row.values.reshape(1, -1)
    direction = clf.predict(X_latest)[0]
    pct_change = reg.predict(X_latest)[0]

    prediction = f"🔮 Prediction: {'UP 📈' if direction == 1 else 'DOWN 📉'}\n"
    prediction += f"Estimated % change in close price: {pct_change:.2f}%"
    return prediction

def main():
    ticker = input("Enter a stock ticker (e.g., AAPL, TSLA, NVDA): ").strip().upper()

    print(f"\n📡 Fetching data for {ticker}...")
    raw_data = get_stock_data(ticker)
    if raw_data is None:
        print("❌ Failed to fetch data. Try a different ticker.")
        return

    print("🔧 Building dataset...")
    dataset = build_dataset(raw_data)
    if dataset.empty:
        print("❌ Not enough data to build dataset.")
        return

    print("🧠 Training models...")
    start_time = time.time()
    clf, reg, acc, err, latest_row = train_models(dataset)
    elapsed_time = time.time() - start_time

    algo_accuracy = (dataset['signal_algo'] == dataset['direction']).mean()

    print(f"\n✅ Results for {ticker}")
    print(f"- Your Algorithm Accuracy: {algo_accuracy * 100:.2f}%")
    print(f"- ML Direction Accuracy: {acc * 100:.2f}%")
    print(f"- ML % Change MAE: {err:.2f}%")
    print(f"- Training Time: {elapsed_time:.2f} seconds\n")

    print(predict_next(clf, reg, latest_row))

if __name__ == "__main__":
    main()
