from flask import Flask, render_template, request, jsonify
import pandas as pd
import datetime
from ta import momentum, trend

app = Flask(__name__)

# Path to the filtered stock data file
DATA_FILE = "filtered_stock_data.csv"


# Custom function to handle Macedonian number format (e.g., 2.440,00 -> 2440.00)
def parse_macedonian_price(price_str):
    try:
        # Remove thousand separator (.)
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
            return jsonify([])

        # Return today's data as JSON
        return jsonify(todays_data.to_dict(orient='records'))

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/technicalAnalysis')
def technical_analysis():
    """Render the technical analysis page with issuers."""
    return render_template('technicalAnalysis.html', issuers=issuers)



@app.route('/technicalAnalysis', methods=['POST'])
def analyze():
    """Analyze the selected issuer's stock data."""
    try:
        # Get the selected issuer
        issuer = request.form.get('issuer')

        # Check if the issuer is valid
        if issuer not in issuers:
            return jsonify({'error': 'Invalid issuer selected'}), 400

        # Filter the data for the selected issuer
        issuer_data = df[df['Issuer Name'] == issuer].sort_values(by='Date')

        if issuer_data.empty:
            return jsonify({'error': 'No data available for the selected issuer'}), 400

        # Ensure data is sorted by date for calculations
        issuer_data = issuer_data.set_index('Date')

        # Check if 'Average Price' column exists
        if 'Average Price' not in issuer_data.columns:
            return jsonify({'error': "'Average Price' column is missing in the data"}), 400

        # Calculate indicators
        indicators = {}
        close_prices = issuer_data['Average Price']

        # 1. Simple Moving Average (SMA)
        sma_20 = close_prices.rolling(window=20).mean().iloc[-1]
        sma_50 = close_prices.rolling(window=50).mean().iloc[-1]

        # 2. Exponential Moving Average (EMA)
        ema_20 = close_prices.ewm(span=20, adjust=False).mean().iloc[-1]
        ema_50 = close_prices.ewm(span=50, adjust=False).mean().iloc[-1]

        # 3. Relative Strength Index (RSI)
        rsi = momentum.rsi(close_prices, window=14).iloc[-1]

        # 4. Moving Average Convergence Divergence (MACD)
        macd = trend.macd(close_prices).iloc[-1]

        # 5. Bollinger Bands (Upper and Lower)
        rolling_std = close_prices.rolling(window=20).std().iloc[-1]
        bollinger_upper = sma_20 + (rolling_std * 2)
        bollinger_lower = sma_20 - (rolling_std * 2)

        # Check for NaN values in any of the indicators
        insufficient_data = {}
        if pd.isna(sma_20):
            insufficient_data['SMA_20'] = 'Insufficient data for SMA_20'
        if pd.isna(sma_50):
            insufficient_data['SMA_50'] = 'Insufficient data for SMA_50'
        if pd.isna(ema_20):
            insufficient_data['EMA_20'] = 'Insufficient data for EMA_20'
        if pd.isna(ema_50):
            insufficient_data['EMA_50'] = 'Insufficient data for EMA_50'
        if pd.isna(rsi):
            insufficient_data['RSI'] = 'Insufficient data for RSI'
        if pd.isna(macd):
            insufficient_data['MACD'] = 'Insufficient data for MACD'
        if pd.isna(bollinger_upper) or pd.isna(bollinger_lower):
            insufficient_data['Bollinger Bands'] = 'Insufficient data for Bollinger Bands'

        # Bundle all the indicators into a dictionary
        indicators = {
            "SMA_20": sma_20,
            "SMA_50": sma_50,
            "EMA_20": ema_20,
            "EMA_50": ema_50,
            "RSI": rsi,
            "MACD": macd,
            "Bollinger_Upper": bollinger_upper,
            "Bollinger_Lower": bollinger_lower
        }

        # If there is insufficient data, return an error message with the details
        if insufficient_data:
            return render_template('technicalAnalysis.html',
                                   insufficient_data=insufficient_data,
                                   issuer=issuer,
                                   issuers=issuers)

        # Return the indicators and the selected issuer as a response
        return render_template('technicalAnalysis.html',
                               indicators=indicators,
                               issuer=issuer,
                               issuers=issuers)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
