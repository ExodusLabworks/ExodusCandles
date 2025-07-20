import discord
from discord.ext import commands
import asyncio
import logging
import os
from typing import Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bot configuration
intents = discord.Intents.default()
intents.message_content = True  # Required for message content access
bot = commands.Bot(command_prefix='!', intents=intents)

# Import your existing Python modules here
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import io
import plotly.graph_objects as go
from dotenv import load_dotenv  # <-- Add this
load_dotenv()  # <-- And this

# # Logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

import numpy as np

@bot.event
async def on_ready():
    """Event triggered when bot is ready"""
    logger.info(f'{bot.user} has connected to Discord!')
    logger.info(f'Bot is in {len(bot.guilds)} guilds')



@bot.event
async def on_command_error(ctx, error):
    """Global error handler"""
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("❌ Command not found. Use `!help` to see available commands.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"❌ Missing required argument: {error.param.name}")
    elif isinstance(error, commands.BadArgument):
        await ctx.send("❌ Invalid argument provided.")
    else:
        logger.error(f"Unexpected error: {error}")
        await ctx.send("❌ An unexpected error occurred.")

# Basic commands
@bot.command(name='ping')
async def ping(ctx):
    """Simple ping command"""
    latency = round(bot.latency * 1000)
    await ctx.send(f'🏓 Pong! Latency: {latency}ms')

@bot.command(name='status')
async def status(ctx):
    """Bot status information"""
    embed = discord.Embed(title="Bot Status", color=0x00ff00)
    embed.add_field(name="Servers", value=len(bot.guilds), inline=True)
    embed.add_field(name="Users", value=len(bot.users), inline=True)
    embed.add_field(name="Latency", value=f"{round(bot.latency * 1000)}ms", inline=True)
    await ctx.send(embed=embed)

# Stock Analysis Commands
@bot.command(name='stock')
async def stock_analysis(ctx, ticker: str, end_date: str = None, days: int = 30):
    """
    Perform stock analysis using Exodus Analytics
    Usage: !stock <ticker> [end_date] [days]
    Example: !stock AAPL 2024-01-15 60
    """
    try:
        ticker = ticker.upper()
        
        # Use today's date if no end_date provided
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
        
        # Validate date format
        try:
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            await ctx.send("❌ Invalid date format. Use YYYY-MM-DD")
            return
        
        # Send initial message
        message = await ctx.send(f"🔄 Analyzing {ticker} for {days} days ending {end_date}...")
        
        # Get stock data
        result = await asyncio.to_thread(get_stock_data, ticker, end_date, days)
        
        if not result:
            await message.edit(content="❌ Failed to retrieve stock data. Check ticker symbol and date.")
            return
        
        # Create analysis report
        report = await asyncio.to_thread(generate_analysis_report, result, ticker, end_date, days)
        
        # Send report in chunks if it's too long
        if len(report) > 2000:
            chunks = [report[i:i+2000] for i in range(0, len(report), 2000)]
            await message.edit(content=chunks[0])
            for chunk in chunks[1:]:
                await ctx.send(chunk)
        else:
            await message.edit(content=report)
            
    except Exception as e:
        logger.error(f"Error in stock analysis: {e}")
        await ctx.send(f"❌ Error analyzing {ticker}: {str(e)}")

@bot.command(name='trend')
async def trend_analysis(ctx, ticker: str, days: int = 30):
    """
    Get trend analysis for a stock
    Usage: !trend <ticker> [days]
    """
    try:
        ticker = ticker.upper()
        end_date = datetime.today().strftime('%Y-%m-%d')
        
        message = await ctx.send(f"📈 Analyzing trend for {ticker}...")
        
        result = await asyncio.to_thread(get_stock_data, ticker, end_date, days)
        
        if not result:
            await message.edit(content="❌ Failed to retrieve stock data.")
            return
        
        trend_info = bullOrBear(result, days, ticker)
        
        embed = discord.Embed(title=f"{ticker} Trend Analysis", color=0x00ff00)
        embed.add_field(name="Market Trend", value=trend_info, inline=False)
        embed.set_footer(text=f"Analysis period: {days} days")
        
        await message.edit(content="", embed=embed)
        
    except Exception as e:
        logger.error(f"Error in trend analysis: {e}")
        await ctx.send(f"❌ Error analyzing trend: {str(e)}")

@bot.command(name='rsi')
async def rsi_analysis(ctx, ticker: str, days: int = 30):
    """
    Get RSI analysis for a stock
    Usage: !rsi <ticker> [days]
    """
    try:
        ticker = ticker.upper()
        end_date = datetime.today().strftime('%Y-%m-%d')
        
        message = await ctx.send(f"🔄 Calculating RSI for {ticker}...")
        
        result = await asyncio.to_thread(get_stock_data, ticker, end_date, days)
        
        if not result:
            await message.edit(content="❌ Failed to retrieve stock data.")
            return
        
        rsi = RSI_determination(result, days)
        
        # Determine RSI status
        if rsi >= 70:
            rsi_status = "Overbought ⚠️"
            color = 0xff0000
        elif rsi <= 30:
            rsi_status = "Oversold 📉"
            color = 0x00ff00
        else:
            rsi_status = "Normal Range"
            color = 0x0099ff
        
        embed = discord.Embed(title=f"{ticker} RSI Analysis", color=color)
        embed.add_field(name="RSI Value", value=f"{rsi:.2f}", inline=True)
        embed.add_field(name="Status", value=rsi_status, inline=True)
        embed.add_field(name="Interpretation", 
                       value="RSI > 70: Potentially overbought\nRSI < 30: Potentially oversold\n30-70: Normal trading range", 
                       inline=False)
        embed.set_footer(text=f"Analysis period: {days} days")
        
        await message.edit(content="", embed=embed)
        
    except Exception as e:
        logger.error(f"Error in RSI analysis: {e}")
        await ctx.send(f"❌ Error calculating RSI: {str(e)}")

@bot.command(name='compare')
async def compare_stocks(ctx, ticker1: str, ticker2: str, days: int = 30):
    """
    Compare two stocks
    Usage: !compare <ticker1> <ticker2> [days]
    """
    try:
        ticker1 = ticker1.upper()
        ticker2 = ticker2.upper()
        end_date = datetime.today().strftime('%Y-%m-%d')
        
        message = await ctx.send(f"⚖️ Comparing {ticker1} vs {ticker2}...")
        
        # Get data for both stocks
        result1 = await asyncio.to_thread(get_stock_data, ticker1, end_date, days)
        result2 = await asyncio.to_thread(get_stock_data, ticker2, end_date, days)
        
        if not result1 or not result2:
            await message.edit(content="❌ Failed to retrieve data for one or both stocks.")
            return
        
        # Calculate metrics for both
        rsi1 = RSI_determination(result1, days)
        rsi2 = RSI_determination(result2, days)
        
        sma1 = SMA_(result1, days)
        sma2 = SMA_(result2, days)
        
        current1 = result1[2][-1]  # Latest close
        current2 = result2[2][-1]
        
        embed = discord.Embed(title=f"{ticker1} vs {ticker2} Comparison", color=0x9932cc)
        
        embed.add_field(name=f"{ticker1}", 
                       value=f"Price: ${current1:.2f}\nRSI: {rsi1:.2f}\nSMA: ${sma1:.2f}", 
                       inline=True)
        
        embed.add_field(name=f"{ticker2}", 
                       value=f"Price: ${current2:.2f}\nRSI: {rsi2:.2f}\nSMA: ${sma2:.2f}", 
                       inline=True)
        
        embed.add_field(name="Analysis", 
                       value=f"Price ratio: {current1/current2:.2f}\nRSI difference: {abs(rsi1-rsi2):.2f}", 
                       inline=False)
        
        embed.set_footer(text=f"Analysis period: {days} days")
        
        await message.edit(content="", embed=embed)
        
    except Exception as e:
        logger.error(f"Error in comparison: {e}")
        await ctx.send(f"❌ Error comparing stocks: {str(e)}")

@bot.command(name='csv')
async def download_csv(ctx, ticker: str):
    """
    Download OHLCV data as CSV for a ticker
    Usage: !csv <ticker>
    """
    try:
        ticker = ticker.upper()
        
        if not ticker.isalpha():
            await ctx.send("❌ Invalid ticker format. Use letters only (e.g., AAPL, MSFT)")
            return
        
        message = await ctx.send(f"📊 Downloading {ticker} OHLCV data...")
        
        # Generate CSV data
        csv_data = await asyncio.to_thread(get_ohlcv_csv_data, ticker)
        
        if csv_data is None:
            await message.edit(content=f"❌ No data found for {ticker}")
            return
        
        # Create file and send
        filename = f"{ticker}_ohlcv.csv"
        file_obj = io.StringIO(csv_data)
        discord_file = discord.File(file_obj, filename=filename)
        
        await message.edit(content=f"✅ Generated {ticker} OHLCV data for past 60 days")
        await ctx.send(file=discord_file)
        
    except Exception as e:
        logger.error(f"Error downloading CSV: {e}")
        await ctx.send(f"❌ Error downloading {ticker} data: {str(e)}")

@bot.command(name='quick')
async def quick_analysis(ctx, ticker: str):
    """
    Quick stock analysis with key metrics
    Usage: !quick <ticker>
    """
    try:
        ticker = ticker.upper()
        end_date = datetime.today().strftime('%Y-%m-%d')
        
        message = await ctx.send(f"⚡ Quick analysis for {ticker}...")
        
        result = await asyncio.to_thread(get_stock_data, ticker, end_date, 30)
        
        if not result:
            await message.edit(content="❌ Failed to retrieve stock data.")
            return
        
        # Generate quick summary
        summary = await asyncio.to_thread(generate_quick_summary, result, ticker, end_date)
        
        embed = discord.Embed(title=f"{ticker} Quick Analysis", color=0x00ff00)
        embed.description = summary
        embed.set_footer(text=f"Data as of {end_date}")
        
        await message.edit(content="", embed=embed)
        
    except Exception as e:
        logger.error(f"Error in quick analysis: {e}")
        await ctx.send(f"❌ Error analyzing {ticker}: {str(e)}")

@bot.command(name='chart')
async def candlestick_chart(ctx, ticker: str, days: int = 60):
    """
    Generate candlestick chart with EMA, SMA, and bull/bear line
    Usage: !chart <ticker> [days]
    """
    try:
        ticker = ticker.upper()
        
        if days > 252:  # Limit to 1 year of data
            days = 252
        elif days < 10:
            days = 10
        
        message = await ctx.send(f"📊 Generating candlestick chart for {ticker}...")
        
        # Get stock data
        end_date = datetime.today().strftime('%Y-%m-%d')
        result = await asyncio.to_thread(get_stock_data, ticker, end_date, days)
        
        if not result:
            await message.edit(content="❌ Failed to retrieve stock data.")
            return
        
        # Generate chart
        chart_buffer = await asyncio.to_thread(create_candlestick_chart, result, ticker, days)
        
        if chart_buffer is None:
            await message.edit(content="❌ Failed to generate chart.")
            return
        
        # Send chart as file
        filename = f"{ticker}_candlestick_chart.png"
        discord_file = discord.File(chart_buffer, filename=filename)
        
        await message.edit(content=f"📈 {ticker} Candlestick Chart ({days} days)")
        await ctx.send(file=discord_file)
        
    except Exception as e:
        logger.error(f"Error generating chart: {e}")
        await ctx.send(f"❌ Error generating chart: {str(e)}")

@bot.command(name='list')
async def list_operations(ctx):
    """List available operations"""
    operations = [
    "stock <ticker> [end_date] [days] - Full stock analysis",
    "chart <ticker> [days] - Candlestick chart with indicators",
    "csv <ticker> - Download OHLCV data as CSV",
    "quick <ticker> - Quick analysis with key metrics",
    "trend <ticker> [days] - Trend analysis (Bull/Bear)",
    "rsi <ticker> [days] - RSI analysis",
    "compare <ticker1> <ticker2> [days] - Compare two stocks",
    "pred <ticker> [days] - Predict stock movement (Exodus algo)"
]
    
    embed = discord.Embed(title="📈 Stock Analysis Bot Commands", color=0x0099ff)
    embed.description = "\n".join(f"• **!{op}**" for op in operations)
    embed.add_field(name="Examples", 
                   value="!stock AAPL\n!chart TSLA 30\n!csv MSFT\n!quick NVDA\n!rsi AAPL 60\n!compare AAPL MSFT", 
                   inline=False)
    embed.add_field(name="Notes", 
                   value="• Default analysis period is 30-60 days\n• Dates should be in YYYY-MM-DD format\n• All tickers should be NYSE/NASDAQ symbols", 
                   inline=False)
    await ctx.send(embed=embed)

# Admin commands (optional)
@bot.command(name='shutdown')
@commands.has_permissions(administrator=True)
async def shutdown(ctx):
    """Shutdown the bot (admin only)"""
    await ctx.send("🔄 Shutting down...")
    await bot.close()

# Stock Analysis Functions (Your existing code with fixes)
def get_stock_data(ticker, end_date, num_trading_days):
    """
    Fixed version of get_stock_data that handles data properly
    """
    try:
        # Convert end date to datetime
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Use a more generous buffer for trading days
        buffer_days = num_trading_days * 2  # Increased buffer
        start_dt = end_dt - timedelta(days=buffer_days)
        start_date = start_dt.strftime("%Y-%m-%d")
        
        print(f"Requesting data for {ticker} from {start_date} to {end_date}")
        
        # Download data with proper error handling
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None
        
        if stock_data.empty:
            print("No data found. Please check the ticker or date range.")
            return None
        
        # Handle both single and multi-ticker downloads
        if isinstance(stock_data.columns, pd.MultiIndex):
            # Multi-ticker case (shouldn't happen with single ticker, but just in case)
            stock_data.columns = [col[0] for col in stock_data.columns]
        
        # Extract columns (yfinance returns standard column names for single ticker)
        opens = stock_data["Open"].dropna().astype(float).tolist()
        closes = stock_data["Close"].dropna().astype(float).tolist()
        highs = stock_data["High"].dropna().astype(float).tolist()
        lows = stock_data["Low"].dropna().astype(float).tolist()
        volumes = stock_data["Volume"].dropna().astype(int).tolist()
        dates = stock_data.index.strftime('%Y-%m-%d').tolist()
        
        # Ensure we have enough data
        if len(closes) < num_trading_days:
            print(f"Warning: Only {len(closes)} days of data available, requested {num_trading_days}")
        
        return dates, opens, closes, highs, lows, volumes
        
    except Exception as e:
        print(f"An error occurred in get_stock_data: {e}")
        return None

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

        signal_algo = 1 if phi > sigma else 0
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

def train_and_predict(ticker: str):
    raw_data = get_stock_data(ticker)
    if raw_data is None:
        return None, "Failed to fetch stock data."

    dataset = build_dataset(raw_data)
    if dataset.empty:
        return None, "Not enough data to train model."

    features = ['open', 'high', 'low', 'close', 'volume', 'fib', 'sma', 'ema', 'sigma', 'phi', 'signal_algo']
    X = dataset[features]
    y_dir = dataset['direction']
    y_pct = dataset['pct_change']

    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y_dir, test_size=0.2, random_state=42)
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_pct, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    reg = RandomForestRegressor(n_estimators=100, random_state=42)

    clf.fit(X_train_d, y_train_d)
    reg.fit(X_train_p, y_train_p)

    direction_pred = clf.predict(X.iloc[[-1]])[0]
    pct_change_pred = reg.predict(X.iloc[[-1]])[0]

    ml_accuracy = accuracy_score(y_test_d, clf.predict(X_test_d))
    ml_mae = mean_absolute_error(y_test_p, reg.predict(X_test_p))
    algo_accuracy = (dataset['signal_algo'] == dataset['direction']).mean()

    result = {
        "ticker": ticker,
        "direction": "UP 📈" if direction_pred == 1 else "DOWN 📉",
        "pct_change": round(pct_change_pred, 2),
        "ml_accuracy": round(ml_accuracy * 100, 2),
        "ml_mae": round(ml_mae, 2),
        "algo_accuracy": round(algo_accuracy * 100, 2),
    }

    return result, None

def bullOrBear(result, d, ticker):
    """
    Fixed bull/bear determination with better error handling
    """
    try:
        closes = result[2]  # Close prices
        if len(closes) < 2:
            return f"Insufficient data for {ticker}"
        
        start_value = closes[0]
        end_value = closes[-1]
        
        change_percent = ((end_value - start_value) / start_value) * 100
        
        if start_value > end_value:
            return f"{ticker} for {d} days is experiencing a BEAR market (-{abs(change_percent):.2f}%)"
        elif end_value > start_value:
            return f"{ticker} for {d} days is experiencing a BULL market (+{change_percent:.2f}%)"
        else:
            return f"{ticker} for {d} days is experiencing minimal change ({change_percent:.2f}%)"
    except Exception as e:
        return f"Error calculating trend for {ticker}: {str(e)}"

def RSI_determination(result, d):
    """
    Fixed RSI calculation with proper error handling
    """
    try:
        closes = result[2]
        if len(closes) < 15:  # Need at least 15 data points for RSI
            return 50.0  # Return neutral RSI if insufficient data
        
        # Calculate price changes
        price_changes = []
        for i in range(1, len(closes)):
            price_changes.append(closes[i] - closes[i-1])
        
        # Calculate gains and losses for last 14 periods
        gains = []
        losses = []
        
        for change in price_changes[-14:]:  # Last 14 changes
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14
        
        if avg_loss == 0:
            return 100.0  # All gains, max RSI
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    except Exception as e:
        print(f"Error calculating RSI: {e}")
        return 50.0  # Return neutral RSI on error

def SMA_(result, d):
    """
    Fixed Simple Moving Average calculation
    """
    try:
        closes = result[2]
        if len(closes) < d:
            d = len(closes)  # Use all available data if less than requested period
        
        recent_closes = closes[-d:]  # Get last d closes
        sma = sum(recent_closes) / len(recent_closes)
        return sma
        
    except Exception as e:
        print(f"Error calculating SMA: {e}")
        return 0.0

def EMA_SMA(result, d):
    """
    Fixed Exponential Moving Average calculation
    """
    try:
        closes = result[2]
        if len(closes) < d:
            return SMA_(result, len(closes))  # Fall back to SMA if insufficient data
        
        # Calculate SMA for initial EMA
        sma = SMA_(result, d)
        
        # EMA multiplier
        multiplier = 2 / (d + 1)
        
        # Calculate EMA
        ema = sma
        for i in range(len(closes) - d, len(closes)):
            ema = (closes[i] * multiplier) + (ema * (1 - multiplier))
        
        return ema
        
    except Exception as e:
        print(f"Error calculating EMA: {e}")
        return SMA_(result, d)

def get_ohlcv_csv_data(ticker):
    """Generate CSV data for OHLCV"""
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=90)  # Get more data to ensure 60 trading days
        
        stock = yf.Ticker(ticker)
        ohlcv = stock.history(start=start_date, end=end_date)
        
        if ohlcv.empty:
            return None
        
        # Keep only the last 60 trading days
        ohlcv = ohlcv.tail(60)
        
        # Remove timezone information from the index if present
        if hasattr(ohlcv.index, 'tz') and ohlcv.index.tz is not None:
            ohlcv.index = ohlcv.index.tz_localize(None)
        
        ohlcv_filtered = ohlcv[["Open", "High", "Low", "Close", "Volume"]]
        ohlcv_filtered.index.name = "Date"
        
        # Convert to CSV string
        return ohlcv_filtered.to_csv()
        
    except Exception as e:
        print(f"Error downloading OHLCV data: {e}")
        return None

def generate_analysis_report(result, ticker, end_date, days):
    """Generate comprehensive analysis report"""
    try:
        if not result:
            return "❌ No data available for analysis"
        
        dates, opens, closes, highs, lows, volumes = result
        
        if len(closes) == 0:
            return "❌ No price data available"
        
        # Calculate metrics
        market_trend = bullOrBear(result, days, ticker)
        rsi = RSI_determination(result, days)
        sma = SMA_(result, days)
        ema = EMA_SMA(result, days)
        
        # Current price info
        current_price = closes[-1]
        prev_price = closes[-2] if len(closes) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
        
        # Build report
        report = f"📊 **{ticker} Analysis Report**\n"
        report += f"📅 Period: {days} days ending {end_date}\n"
        report += f"💹 Current Price: ${current_price:.2f}\n"
        report += f"📈 Daily Change: ${price_change:+.2f} ({price_change_pct:+.2f}%)\n\n"
        
        report += f"🔍 **Market Trend**\n{market_trend}\n\n"
        
        report += f"📈 **Technical Indicators**\n"
        report += f"RSI: {rsi:.2f} "
        if rsi >= 70:
            report += "(Overbought ⚠️)\n"
        elif rsi <= 30:
            report += "(Oversold 📉)\n"
        else:
            report += "(Normal Range)\n"
        
        report += f"SMA ({days}d): ${sma:.2f}\n"
        report += f"EMA ({days}d): ${ema:.2f}\n\n"
        
        # Price vs moving averages
        report += f"📊 **Price Analysis**\n"
        if current_price > sma:
            report += f"Price is **{((current_price/sma-1)*100):.1f}%** above SMA (Bullish 📈)\n"
        else:
            report += f"Price is **{((1-current_price/sma)*100):.1f}%** below SMA (Bearish 📉)\n"
        
        if current_price > ema:
            report += f"Price is **{((current_price/ema-1)*100):.1f}%** above EMA (Bullish 📈)\n"
        else:
            report += f"Price is **{((1-current_price/ema)*100):.1f}%** below EMA (Bearish 📉)\n"
        
        return report
        
    except Exception as e:
        return f"❌ Error generating report: {str(e)}"

# Candlestick chart using Plotly
def create_candlestick_chart(result, ticker, days):
    try:
        dates, opens, closes, highs, lows, volumes = result
        df = pd.DataFrame({
            'Date': pd.to_datetime(dates),
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes
        })
        df['SMA'] = df['Close'].rolling(window=20).mean()
        df['EMA'] = df['Close'].ewm(span=20, adjust=False).mean()

        fig = go.Figure(data=[
            go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                           low=df['Low'], close=df['Close'], name='Candles'),
            go.Scatter(x=df['Date'], y=df['SMA'], mode='lines', name='SMA (20)', line=dict(color='purple')),
            go.Scatter(x=df['Date'], y=df['EMA'], mode='lines', name='EMA (20)', line=dict(color='blue'))
        ])

        fig.update_layout(
            title=f"{ticker} Candlestick Chart with SMA & EMA",
            yaxis_title='Price ($)',
            xaxis_title='Date',
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            width=1000,
            height=600
        )

        img_bytes = fig.to_image(format="png")
        buf = io.BytesIO(img_bytes)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Error creating chart with Plotly: {e}")
        return None
    



def build_dataset(result):
    highs = result[3]
    lows = result[4]
    closes = result[2]
    opens = result[1]
    volumes = result[5]
    
    data = []
    for i in range(3, len(closes) - 1):  # Leave 1 for t+1 close
        open_price = opens[i]
        close_price = closes[i]
        high = highs[i]
        low = lows[i]
        volume = volumes[i]

        median = (open_price + close_price) / 2
        fib = sum(closes[i-3:i]) / 3
        sma = sum(closes[i-3:i]) / 3
        ema = closes[i] * 0.5 + closes[i-1] * 0.3 + closes[i-2] * 0.2  # rough EMA
        
        # Wick calculations
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
        
        # Target: % change of next day
        next_close = closes[i+1]
        pct_change = ((next_close - close_price) / close_price) * 100
        
        data.append({
            'open': open_price,
            'close': close_price,
            'high': high,
            'low': low,
            'volume': volume,
            'fib': fib,
            'sma': sma,
            'ema': ema,
            'sigma': sigma,
            'phi': phi,
            'median': median,
            'pct_change': pct_change,
            'direction': 1 if pct_change > 0 else 0
        })

    return pd.DataFrame(data)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

def train_models(df):
    X = df[['open', 'high', 'low', 'close', 'volume', 'fib', 'sma', 'ema', 'sigma', 'phi']]
    
    # Direction model (classifier)
    y_dir = df['direction']
    X_train, X_test, y_train, y_test = train_test_split(X, y_dir, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    # % Change model (regressor)
    y_pct = df['pct_change']
    X_train, X_test, y_train, y_test = train_test_split(X, y_pct, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=100)
    reg.fit(X_train, y_train)
    error = mean_absolute_error(y_test, reg.predict(X_test))

    return clf, reg, acc, error

def predict_next(clf, reg, latest_row):
    features = latest_row[['open', 'high', 'low', 'close', 'volume', 'fib', 'sma', 'ema', 'sigma', 'phi']].values.reshape(1, -1)
    direction = clf.predict(features)[0]
    pct_change = reg.predict(features)[0]

    result = f"🔮 Prediction: Stock will likely {'INCREASE' if direction == 1 else 'DECREASE'} tomorrow\n"
    result += f"Estimated % Change: {pct_change:.2f}%"
    return result


def prediction(result, d, ticker, end_date):
    highs = result[3]
    lows = result[4]
    closes = result[2]
    opens = result[1]
    medians, fibs, ratios, alphas, betas, sigma, phi = [], [], [], [], [], [], []
    accuracy = 0

    for i in range(len(highs)):
        medians.append(abs((opens[i] + closes[i]) / 2))
        if opens[i] >= closes[i]:
            alphas.append(highs[i])
            betas.append(lows[i])
        else:
            alphas.append(lows[i])
            betas.append(highs[i])
    
    for i in range(len(highs)):
        if opens[i] >= closes[i]:
            sigma.append(alphas[i] - opens[i])
            phi.append(closes[i] - betas[i])
        else:
            sigma.append(betas[i] - closes[i])
            phi.append(opens[i] - alphas[i])
    
    for i in range(len(opens)):
        if i < 3:
            fibs.append(closes[i])
        else:
            fibs.append((closes[i] + closes[i-1] + closes[i-2]) / 3)
    
    for i in range(len(fibs)):
        if i == 0:
            ratios.append("0")
        else:
            if fibs[i] > fibs[i-1]:
                ratios.append("+")
            elif fibs[i] < fibs[i-1]:
                ratios.append("-")
            else:
                ratios.append("0")
    
    for i in range(len(sigma) - 1):
        if sigma[i] > 0.2 and phi[i] < 0.2:
            ratios[i+1] = "-"
        elif sigma[i] < 0.2 and phi[i] > 0.2:
            ratios[i+1] = "+"
    
    for i in range(len(ratios) - 1):
        if ratios[i] == "+" and (medians[i+1] > medians[i]):
            accuracy += 1
        elif ratios[i] == "-" and (medians[i+1] < medians[i]):
            accuracy += 1

    prediction_accuracy = (accuracy / len(ratios)) * 100
    next_day_prediction = ""

    if prediction_accuracy >= 50:
        if ratios[d - 1] == "+":
            next_day_prediction = f"{ticker} will likely close HIGHER tomorrow than {end_date}"
        else:
            next_day_prediction = f"{ticker} will likely close LOWER tomorrow than {end_date}"
    else:
        if ratios[d - 1] == "+":
            next_day_prediction = f"{ticker} will likely close LOWER tomorrow than {end_date} ::: INVERSE USED"
        else:
            next_day_prediction = f"{ticker} will likely close HIGHER tomorrow than {end_date} ::: INVERSE USED"
    
    return f"📉 **Prediction for {ticker}**\n" \
           f"Period analyzed: {d} days ending {end_date}\n" \
           f"Prediction: {next_day_prediction}\n" \
           f"Prediction accuracy: {prediction_accuracy:.2f}%"

@bot.command(name='algo')
async def prediction_command(ctx, ticker: str, days: int = 30):
    """
    Predict stock direction using Fibonacci-based wick algorithm
    Usage: !pred <ticker> [days] or !algo <ticker> [days]
    """
    try:
        ticker = ticker.upper()
        end_date = datetime.today().strftime('%Y-%m-%d')
        
        message = await ctx.send(f"🔮 Running prediction algorithm for {ticker}...")
        
        result = await asyncio.to_thread(get_stock_data, ticker, end_date, days)
        
        if not result:
            await message.edit(content=f"❌ Failed to retrieve data for {ticker}")
            return
        
        prediction_result = await asyncio.to_thread(prediction, result, days, ticker, end_date)
        
        await message.edit(content=prediction_result)
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        await ctx.send(f"❌ Error running prediction for {ticker}: {str(e)}")

# === Predict Function (from your algorithm) ===
def predict_stock(ticker: str):
    # 1. Download stock data
    end_date = datetime.today()
    start_date = end_date - timedelta(days=120)
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty or len(df) < 60:
        return None, "Not enough data"

    df = df.tail(60)
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
        sma = fib
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

        signal_algo = 1 if phi > sigma else 0
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

    dataset = pd.DataFrame(data)
    if dataset.empty:
        return None, "Failed to prepare dataset."

    features = ['open', 'high', 'low', 'close', 'volume', 'fib', 'sma', 'ema', 'sigma', 'phi', 'signal_algo']
    X = dataset[features]
    y_dir = dataset['direction']
    y_pct = dataset['pct_change']

    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y_dir, test_size=0.2, random_state=42)
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_pct, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    reg = RandomForestRegressor(n_estimators=100, random_state=42)

    clf.fit(X_train_d, y_train_d)
    reg.fit(X_train_p, y_train_p)

    direction_pred = clf.predict(X.iloc[[-1]])[0]
    pct_change_pred = reg.predict(X.iloc[[-1]])[0]

    ml_accuracy = accuracy_score(y_test_d, clf.predict(X_test_d))
    ml_mae = mean_absolute_error(y_test_p, reg.predict(X_test_p))
    algo_accuracy = (dataset['signal_algo'] == dataset['direction']).mean()

    result = {
        "ticker": ticker,
        "direction": "UP 📈" if direction_pred == 1 else "DOWN 📉",
        "pct_change": round(pct_change_pred, 2),
        "ml_accuracy": round(ml_accuracy * 100, 2),
        "ml_mae": round(ml_mae, 2),
        "algo_accuracy": round(algo_accuracy * 100, 2),
    }

    return result, None

# === Discord Command ===
@bot.command()
async def pred(ctx, ticker: str):
    await ctx.send(f"🔍 Predicting for `{ticker.upper()}`... Please wait ⏳")

    result, error = predict_stock(ticker)
    if error:
        await ctx.send(f"❌ Error: {error}")
        return

    msg = (
        f"📊 **Prediction for {result['ticker'].upper()}**\n\n"
        f"🔮 Direction: **{result['direction']}**\n"
        f"📈 Estimated % Change: `{result['pct_change']}%`\n\n"
        f"🤖 ML Accuracy: `{result['ml_accuracy']}%`\n"
        f"📉 ML MAE: `{result['ml_mae']}%`\n"
        f"🧠 Algo Accuracy: `{result['algo_accuracy']}%`"
    )

    await ctx.send(msg)
    
def generate_quick_summary(result, ticker, end_date):
    """Generate quick summary for embed"""
    try:
        if not result:
            return "No data available"
        
        dates, opens, closes, highs, lows, volumes = result
        
        if len(closes) == 0:
            return "No price data available"
        
        # Calculate key metrics
        current_price = closes[-1]
        prev_price = closes[-2] if len(closes) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
        
        rsi = RSI_determination(result, len(closes))
        sma = SMA_(result, min(len(closes), 20))  # Use 20-day SMA or available data
        
        summary = f"**Current Price:** ${current_price:.2f}\n"
        summary += f"**Daily Change:** ${price_change:+.2f} ({price_change_pct:+.2f}%)\n"
        summary += f"**RSI:** {rsi:.2f}\n"
        summary += f"**SMA:** ${sma:.2f}\n"
        
        # Trend
        if current_price > sma:
            summary += f"**Trend:** Above SMA - Bullish 📈"
        else:
            summary += f"**Trend:** Below SMA - Bearish 📉"
        
        return summary
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Error handling for missing permissions
@shutdown.error
async def admin_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("❌ You don't have permission to use this command.")

def main():
    """Main function to run the bot"""
    # Get token from environment variable for security
    token = os.getenv('DISCORD_BOT_TOKEN')
    
    if not token:
        logger.error("DISCORD_BOT_TOKEN environment variable not set!")
        logger.error("Please set your bot token: export DISCORD_BOT_TOKEN='your_token_here'")
        return
    
    try:
        bot.run(token)
    except discord.LoginFailure:
        logger.error("Invalid bot token!")
    except Exception as e:
        logger.error(f"Error starting bot: {e}")

# Note: You'll need to install matplotlib if you haven't already:
# pip install matplotlib

if __name__ == "__main__":
    main()




