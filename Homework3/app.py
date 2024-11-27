import os
import sys
from flask import Flask, render_template, jsonify
import pandas as pd
from datetime import datetime

# Add the root directory to sys.path so Homework1 can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from Homework1
from mainFilter import fetch_and_filter_missing_data

app = Flask(__name__)

# Adjust the path to the CSV file in Homework1
csv_file_path = os.path.join(os.path.dirname(__file__), '..', 'Homework1', 'filtered_stock_data.csv')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_todays_data', methods=['GET'])
def get_todays_data():
    try:
        # Call the filter function from mainFilter.py to ensure fresh data
        fetch_and_filter_missing_data()

        # Load the filtered CSV file that mainFilter.py generated
        df = pd.read_csv(csv_file_path, parse_dates=['Date'], dayfirst=True)

        # Filter today's stock exchanges
        today = datetime.now().date()
        todays_data = df[df['Date'].dt.date == today]

        # If no data is found, return an empty result
        if todays_data.empty:
            return jsonify([])

        # Convert DataFrame to JSON
        data = todays_data.to_dict(orient='records')
        return jsonify(data)

    except Exception as e:
        return str(e), 500


if __name__ == '__main__':
    app.run(debug=True)
