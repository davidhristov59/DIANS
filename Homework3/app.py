import os
import datetime
import pandas as pd
from flask import Flask, render_template, request, jsonify
from ta import momentum, trend
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
app = Flask(__name__)

# Path to the filtered stock data file
DATA_FILE = "../Homework1/filtered_stock_data.csv"

# Path to save generated graphs
GRAPH_FOLDER = os.path.join('static', 'graphs')
if not os.path.exists(GRAPH_FOLDER):
    os.makedirs(GRAPH_FOLDER)


# Custom function to handle Macedonian number format (e.g., 2.440,00 -> 2440.00)
def parse_macedonian_price(price_str):
    try:
        price_str = price_str.replace('.', '').replace(',', '.')
        return float(price_str)
    except ValueError:
        return None  # Return None if parsing fails


# Load and preprocess stock data
try:
    df = pd.read_csv(DATA_FILE, dtype={'Date': 'object'})
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
    df['Average Price'] = df['Average Price'].apply(parse_macedonian_price)
    df['Max.'] = df['Max.'].apply(parse_macedonian_price)
    df['Min.'] = df['Min.'].apply(parse_macedonian_price)
    df['Last Transaction Price'] = df['Last Transaction Price'].apply(parse_macedonian_price)
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
        df = pd.read_csv(DATA_FILE, dtype={'Date': 'object'})
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
        df['Average Price'] = df['Average Price'].apply(parse_macedonian_price)

        today = datetime.datetime.now().date()
        todays_data = df[df['Date'].dt.date == today]

        if todays_data.empty:
            return jsonify([])

        todays_data['Date'] = todays_data['Date'].apply(
            lambda d: d.replace(hour=15, minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')
        )
        return jsonify(todays_data.to_dict(orient='records'))

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/technicalAnalysis', methods=['GET', 'POST'])
def technical_analysis():
    """Render and process technical analysis."""
    if request.method == 'GET':
        return render_template('technicalAnalysis.html', issuers=issuers)

    try:
        issuer = request.form.get('issuer')
        if issuer not in issuers:
            return render_template('technicalAnalysis.html', issuers=issuers, error="Invalid issuer selected.")

        issuer_data = df[df['Issuer Name'] == issuer].sort_values(by='Date')
        if issuer_data.empty:
            return render_template('technicalAnalysis.html', issuers=issuers,
                                   error="No data available for the selected issuer.")

        close_prices = issuer_data.set_index('Date')['Average Price']

        indicators = {
            "SMA_20": close_prices.rolling(window=20).mean().iloc[-1] if len(close_prices) >= 20 else None,
            "SMA_50": close_prices.rolling(window=50).mean().iloc[-1] if len(close_prices) >= 50 else None,
            "EMA_20": close_prices.ewm(span=20, adjust=False).mean().iloc[-1] if len(close_prices) >= 20 else None,
            "EMA_50": close_prices.ewm(span=50, adjust=False).mean().iloc[-1] if len(close_prices) >= 50 else None,
            "RSI": momentum.rsi(close_prices, window=14).iloc[-1] if len(close_prices) >= 14 else None,
            "MACD": trend.macd(close_prices).iloc[-1] if len(close_prices) >= 26 else None,
        }
        rolling_std = close_prices.rolling(window=20).std().iloc[-1] if len(close_prices) >= 20 else None
        if rolling_std is not None:
            indicators["Bollinger_Upper"] = indicators["SMA_20"] + 2 * rolling_std
            indicators["Bollinger_Lower"] = indicators["SMA_20"] - 2 * rolling_std

        insufficient_data = {k: "Insufficient data" for k, v in indicators.items() if v is None}
        if insufficient_data:
            return render_template('technicalAnalysis.html', issuers=issuers, insufficient_data=insufficient_data,
                                   issuer=issuer)

        return render_template('technicalAnalysis.html', issuers=issuers, indicators=indicators, issuer=issuer)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def plot_average_price_chart(issuer, df):
    """Generate and save a line chart for the selected issuer's average price over time."""
    issuer_data = df[df['Issuer Name'] == issuer].copy()  # Use .copy() to avoid modifying a view

    if issuer_data.empty:
        return None

    # Ensure that the 'Average Price' column is numeric using .loc to avoid SettingWithCopyWarning
    issuer_data.loc[:, 'Average Price'] = pd.to_numeric(issuer_data['Average Price'], errors='coerce')

    # Drop rows with NaN values in the 'Average Price' column using .loc
    issuer_data.dropna(subset=['Average Price'], inplace=True)

    # Prepare the data for the line chart
    issuer_data = issuer_data[['Date', 'Average Price']]
    issuer_data.set_index('Date', inplace=True)

    # Generate the line chart for Average Price over time
    plt.figure(figsize=(10, 6))
    plt.plot(issuer_data.index, issuer_data['Average Price'], label=f'{issuer} Average Price', color='blue')

    # Formatting the chart
    plt.title(f'{issuer} Average Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Price')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    # Save the graph as a PNG image in the static folder
    graph_filename = f"{issuer}_average_price_chart.png"
    graph_path = os.path.join(GRAPH_FOLDER, graph_filename)
    plt.savefig(graph_path)  # Save the figure instead of showing it in a GUI
    plt.close()  # Close the plot to avoid Matplotlib GUI warning

    return f'graphs/{graph_filename}'

@app.route('/graphs', methods=['GET', 'POST'])
def show_graphs():
    """Render the graphs page and handle graph generation."""
    if request.method == 'POST':
        # Get the selected issuer
        issuer = request.form.get('issuer')

        # Validate issuer selection
        if issuer not in issuers:
            return render_template('graphs.html', issuers=issuers, error="Invalid issuer selected.")

        # Generate the Average Price chart
        graph_path = plot_average_price_chart(issuer, df)

        if graph_path:
            return render_template('graphs.html', issuers=issuers, graph_path=graph_path, selected_issuer=issuer)
        else:
            return render_template('graphs.html', issuers=issuers, error="No data available for the selected issuer.")

    return render_template('graphs.html', issuers=issuers)


if __name__ == '__main__':
    app.run(debug=True)
