import pandas as pd
import datetime
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import matplotlib.pyplot as plt
import io
import base64
import csv

DATA_FILE = "filtered_stock_data.csv"


def load_stock_data():
    """Load the stock data from the CSV file."""
    df = pd.read_csv(DATA_FILE, dtype={'Date': 'object'})
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
    df['Average Price'] = df['Average Price'].apply(parse_macedonian_price)
    return df


def get_issuers(df):
    """Fetch unique issuers from the data."""
    return df['Issuer Name'].unique()


def get_todays_data(df):
    """Return today's stock data, or the last available data."""
    today = datetime.datetime.now().date()
    todays_data = df[df['Date'].dt.date == today]
    if todays_data.empty:
        last_available_date = df['Date'].max().date()
        todays_data = df[df['Date'].dt.date == last_available_date]
    return todays_data


def calculate_technical_indicators(issuer_data):
    """Calculate all required technical indicators for the selected issuer."""
    close_prices = issuer_data['Average Price']

    # RSI
    rsi_indicator = RSIIndicator(close=close_prices, window=14)
    rsi = rsi_indicator.rsi()

    # MACD
    macd_indicator = MACD(close=close_prices)
    macd = macd_indicator.macd()
    macd_signal = macd_indicator.macd_signal()
    macd_diff = macd_indicator.macd_diff()

    # Bollinger Bands
    bollinger = BollingerBands(close=close_prices, window=20, window_dev=2)
    bollinger_upper = bollinger.bollinger_hband()
    bollinger_lower = bollinger.bollinger_lband()

    # Simple and Exponential Moving Averages (SMA, EMA)
    sma_20 = close_prices.rolling(window=20).mean()
    sma_50 = close_prices.rolling(window=50).mean()
    ema_20 = close_prices.ewm(span=20, adjust=False).mean()
    ema_50 = close_prices.ewm(span=50, adjust=False).mean()

    return rsi, macd, macd_signal, macd_diff, bollinger_upper, bollinger_lower, sma_20, sma_50, ema_20, ema_50


def predict_price(indicator, days_ahead):
    """Predict future price based on indicator."""
    last_value = indicator.iloc[-1]
    return last_value * (1 + (days_ahead * 0.001))


def generate_prediction_data(sma_20, sma_50, ema_20, ema_50, rsi, macd, bollinger_upper, bollinger_lower):
    """Generate predicted prices for various time periods (1 day, 1 week, 1 month)."""
    return {
        '1_day': {
            'SMA_20': predict_price(sma_20, 1),
            'SMA_50': predict_price(sma_50, 1),
            'EMA_20': predict_price(ema_20, 1),
            'EMA_50': predict_price(ema_50, 1),
            'RSI': predict_price(rsi, 1),
            'MACD': predict_price(macd, 1),
            'Bollinger_Upper': predict_price(bollinger_upper, 1),
            'Bollinger_Lower': predict_price(bollinger_lower, 1),
        },
        '1_week': {
            'SMA_20': predict_price(sma_20, 7),
            'SMA_50': predict_price(sma_50, 7),
            'EMA_20': predict_price(ema_20, 7),
            'EMA_50': predict_price(ema_50, 7),
            'RSI': predict_price(rsi, 7),
            'MACD': predict_price(macd, 7),
            'Bollinger_Upper': predict_price(bollinger_upper, 7),
            'Bollinger_Lower': predict_price(bollinger_lower, 7),
        },
        '1_month': {
            'SMA_20': predict_price(sma_20, 30),
            'SMA_50': predict_price(sma_50, 30),
            'EMA_20': predict_price(ema_20, 30),
            'EMA_50': predict_price(ema_50, 30),
            'RSI': predict_price(rsi, 30),
            'MACD': predict_price(macd, 30),
            'Bollinger_Upper': predict_price(bollinger_upper, 30),
            'Bollinger_Lower': predict_price(bollinger_lower, 30),
        }
    }


def generate_graph(close_prices, sma_20, sma_50, ema_20, ema_50, bollinger_upper, bollinger_lower, issuer):
    """Generate and return the graph for the selected issuer as base64."""
    plt.figure(figsize=(10, 6))
    plt.plot(close_prices, label='Average Price', color='blue', linestyle='-', linewidth=2)
    plt.plot(sma_20, label='SMA 20', linestyle='--', color='orange')
    plt.plot(sma_50, label='SMA 50', linestyle='--', color='red')
    plt.plot(ema_20, label='EMA 20', linestyle='-', color='green')
    plt.plot(ema_50, label='EMA 50', linestyle='-', color='purple')
    plt.plot(bollinger_upper, label='Bollinger Upper', linestyle=':', color='magenta')
    plt.plot(bollinger_lower, label='Bollinger Lower', linestyle=':', color='cyan')

    plt.title(f'Technical Analysis for {issuer}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    graph_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    plt.close()

    return graph_base64

def parse_macedonian_price(price_str):
    """Convert Macedonian-formatted price string to float."""
    try:
        price_str = price_str.replace('.', '').replace(',', '.')
        return float(price_str)
    except ValueError:
        return None


def get_description_for_issuer(issuer):
    try:
        with open('../prediction_service/data/analysis_results.csv', mode='r', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                if row['Issuer Name'] == issuer:
                    return row['Description']
    except Exception as e:
        print(f"Error reading analysis_results.csv: {e}")
    return "Описот не е достапен."  # Default message when no description is found


def get_short_term_predictions(csv_file):
    df = pd.read_csv(csv_file)
    predictions = []

    for index, row in df.iterrows():
        predictions.append({
            'Issuer': row['Issuer'],
            '1 Week': row['1 Week'],
            '1 Month': row['1 Month'],
            '3 Months': row['3 Months']
        })

    return predictions


def get_medium_term_predictions(csv_file):
    df = pd.read_csv(csv_file)
    predictions = []

    for index, row in df.iterrows():
        predictions.append({
            'Issuer': row['Issuer'],
            '6 Months': row['6 Months'],
            '9 Months': row['9 Months'],
            '1 Year': row['1 Year']
        })

    return predictions


def create_graph(issuer, prices, prediction_type):
    fig, ax = plt.subplots()

    if prediction_type == 'short-term':
        time_frames = ['1 Week', '1 Month', '3 Months']
    else:
        time_frames = ['6 Months', '9 Months', '1 Year']

    ax.plot(time_frames, prices, marker='o', linestyle='-', color='b')
    ax.set_title(f"{issuer} Price Predictions ({prediction_type.capitalize()})")
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Price")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)

    return img_base64
