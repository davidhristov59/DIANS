import os

import matplotlib
from flask import Flask, render_template, request, jsonify
import pandas as pd
import datetime

from matplotlib import pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

matplotlib.use('Agg')
import io
import base64

app = Flask(__name__)

DATA_FILE = "../Homework1/filtered_stock_data.csv"

# Path to save generated graphs
GRAPH_FOLDER = os.path.join('static', 'graphs')
if not os.path.exists(GRAPH_FOLDER):
    os.makedirs(GRAPH_FOLDER)


# Custom function to handle Macedonian number format (e.g., 2.440,00 -> 2440.00)
def parse_macedonian_price(price_str):
    try:
        # Remove the thousand separator (.)
        price_str = price_str.replace('.', '')
        # Replace decimal separator (,) with dot (.)
        price_str = price_str.replace(',', '.')
        return float(price_str)
    except ValueError:
        return None  # Return None if parsing fails


# Load data initially to fetch issuers
try:
    # Read the CSV and treat the 'Date' column as a string (object)
    df = pd.read_csv(DATA_FILE, dtype={'Date': 'object'})

    # Convert 'Date' column to datetime using the specific format
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')

    # Apply the custom parsing to the 'Average Price' column
    df['Average Price'] = df['Average Price'].apply(parse_macedonian_price)

    # Fetch the list of unique issuers
    issuers = df['Issuer Name'].unique()

except Exception as e:
    df = pd.DataFrame()  # Fallback to an empty DataFrame
    issuers = []  # Default to an empty list if loading fails


@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')


@app.route('/get_todays_data', methods=['GET'])
def get_todays_data():
    """Fetch today's stock data."""
    try:
        # Reload the filtered data
        df = pd.read_csv(DATA_FILE, dtype={'Date': 'object'})

        # Convert 'Date' column to datetime using the specific format
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')

        # Apply the custom price parsing again
        df['Average Price'] = df['Average Price'].apply(parse_macedonian_price)

        # Get today's date
        today = datetime.datetime.now().date()

        # Filter today's data
        todays_data = df[df['Date'].dt.date == today]

        if todays_data.empty:
            last_available_date = df['Date'].max().date()
            todays_data = df[df['Date'].dt.date == last_available_date]

        if todays_data.empty:
            return jsonify([])

        todays_data['Date'] = todays_data['Date'].apply(
            lambda d: d.replace(hour=15, minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')
        )

        # Return today's data as JSON
        return jsonify(todays_data.to_dict(orient='records'))

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/technicalAnalysis', methods=['GET', 'POST'])
def technical_analysis():
    """Render or analyze the technical analysis page with issuers."""
    if request.method == 'GET':
        # Render the technical analysis page with available issuers
        return render_template('technicalAnalysis.html', issuers=issuers)

    if request.method == 'POST':
        # Handle the analysis request for a specific issuer
        try:
            issuer = request.form.get('issuer')

            # Check if the issuer is valid
            if issuer not in issuers:
                return jsonify({'error': 'Invalid issuer selected'}), 400

            # Filter data for the selected issuer
            issuer_data = df[df['Issuer Name'] == issuer].sort_values(by='Date')

            if issuer_data.empty:
                return jsonify({'error': 'No data available for the selected issuer'}), 400

            # Ensure data is sorted by date
            issuer_data['Date'] = pd.to_datetime(issuer_data['Date'], errors='coerce')
            issuer_data = issuer_data.set_index('Date')

            # Check if 'Average Price' column exists
            if 'Average Price' not in issuer_data.columns:
                return jsonify({'error': "'Average Price' column is missing in the data"}), 400

            issuer_data['Average Price'] = pd.to_numeric(issuer_data['Average Price'], errors='coerce')
            issuer_data.dropna(subset=['Average Price'], inplace=True)
            issuer_data.sort_index(inplace=True)

            # Calculate technical indicators
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

            # Plot the graph with indicators
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

            # Save the graph as base64
            img_io = io.BytesIO()
            plt.savefig(img_io, format='png')
            img_io.seek(0)
            graph_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
            plt.close()

            # Latest values for the indicators
            indicators = {
                'SMA_20': round(sma_20.iloc[-1], 2) if not pd.isna(sma_20.iloc[-1]) else "Недостаток на податоци",
                'SMA_50': round(sma_50.iloc[-1], 2) if not pd.isna(sma_50.iloc[-1]) else "Недостаток на податоци",
                'EMA_20': round(ema_20.iloc[-1], 2) if not pd.isna(ema_20.iloc[-1]) else "Недостаток на податоци",
                'EMA_50': round(ema_50.iloc[-1], 2) if not pd.isna(ema_50.iloc[-1]) else "Недостаток на податоци",
                'RSI': round(rsi.iloc[-1], 2) if not pd.isna(rsi.iloc[-1]) else "Недостаток на податоци",
                'MACD': round(macd.iloc[-1], 2) if not pd.isna(macd.iloc[-1]) else "Недостаток на податоци",
                'Bollinger_Upper': round(bollinger_upper.iloc[-1], 2) if not pd.isna(
                    bollinger_upper.iloc[-1]) else "Недостаток на податоци",
                'Bollinger_Lower': round(bollinger_lower.iloc[-1], 2) if not pd.isna(
                    bollinger_lower.iloc[-1]) else "Недостаток на податоци",
            }

            return render_template('technicalAnalysis.html',
                                   indicators=indicators,
                                   issuer=issuer,
                                   issuers=issuers,
                                   graph_base64=graph_base64)

        except Exception as e:
            return jsonify({'error': str(e)}), 500


def plot_average_price_chart(issuer, df):
    issuer_data = df[df['Issuer Name'] == issuer].copy()

    if issuer_data.empty:
        return None

    issuer_data['Date'] = pd.to_datetime(issuer_data['Date'], errors='coerce')
    issuer_data.loc[:, 'Average Price'] = pd.to_numeric(issuer_data['Average Price'], errors='coerce')

    issuer_data.dropna(subset=['Date', 'Average Price'], inplace=True)

    issuer_data.set_index('Date', inplace=True)
    issuer_data.sort_index(inplace=True)

    issuer_data['Average Price'] = issuer_data['Average Price'].interpolate()

    # Generate the line chart
    plt.figure(figsize=(10, 6))
    plt.plot(
        issuer_data.index, issuer_data['Average Price'],
        label=f'{issuer} Average Price', color='blue', linestyle='-', linewidth=2
    )
    plt.title(f'{issuer} Просечна Цена Низ Времето')
    plt.xlabel('Date')
    plt.ylabel('Average Price')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return img_base64


@app.route('/about')
def about():
    return render_template('about_us.html')


if __name__ == '__main__':
    app.run(debug=True)
