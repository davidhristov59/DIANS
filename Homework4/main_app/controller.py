from flask import Flask, render_template, request, jsonify
from models import (get_short_term_predictions
, get_medium_term_predictions, create_graph, load_stock_data, get_todays_data
, get_issuers, calculate_technical_indicators, get_description_for_issuer,
generate_prediction_data, generate_graph)
import pandas as pd


df = load_stock_data()  # Load data once during app startup
issuers = get_issuers(df)  # Fetch list of unique issuers

app = Flask(__name__)
@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')


@app.route('/get_todays_data', methods=['GET'])
def get_todays_data_route():
    """Handle request for today's stock data."""
    todays_data = get_todays_data(df)
    if todays_data.empty:
        return jsonify([])
    return jsonify(todays_data.to_dict(orient='records'))


@app.route('/technicalAnalysis', methods=['GET', 'POST'])
def technical_analysis():
    """Render or analyze the technical analysis page with issuers."""
    issuer = None
    predicted_prices = {}
    indicators = {}
    graph_base64 = None

    if request.method == 'POST':
        issuer = request.form.get('issuer')

        if issuer not in issuers:
            return jsonify({'error': 'Invalid issuer selected'}), 400

        # Filter data for the selected issuer
        issuer_data = df[df['Issuer Name'] == issuer].sort_values(by='Date')

        if issuer_data.empty:
            return jsonify({'error': 'No data available for the selected issuer'}), 400

        # Ensure data is sorted by date
        issuer_data['Date'] = pd.to_datetime(issuer_data['Date'], errors='coerce')
        issuer_data = issuer_data.set_index('Date')

        # Calculate technical indicators
        rsi, macd, macd_signal, macd_diff, bollinger_upper, bollinger_lower, sma_20, sma_50, ema_20, ema_50 = calculate_technical_indicators(
            issuer_data)

        # Generate predicted prices
        predicted_prices = generate_prediction_data(sma_20, sma_50, ema_20, ema_50, rsi, macd, bollinger_upper,
                                                    bollinger_lower)

        # Generate the graph
        graph_base64 = generate_graph(issuer_data['Average Price'], sma_20, sma_50, ema_20, ema_50, bollinger_upper,
                                      bollinger_lower, issuer)

        # Prepare indicator values
        indicators = {
            'SMA_20': round(sma_20.iloc[-1], 2),
            'SMA_50': round(sma_50.iloc[-1], 2),
            'EMA_20': round(ema_20.iloc[-1], 2),
            'EMA_50': round(ema_50.iloc[-1], 2),
            'RSI': round(rsi.iloc[-1], 2),
            'MACD': round(macd.iloc[-1], 2),
            'Bollinger_Upper': round(bollinger_upper.iloc[-1], 2),
            'Bollinger_Lower': round(bollinger_lower.iloc[-1], 2),
        }

    return render_template('technicalAnalysis.html',
                           indicators=indicators,
                           issuer=issuer,
                           issuers=issuers,
                           graph_base64=graph_base64,
                           predicted_prices=predicted_prices)


@app.route('/fundamentalAnalysis', methods=['GET', 'POST'])
def fundamental_analysis():
    issuers = ['ALK', 'CKB', 'GRNT', 'KMB', 'MPT', 'MSTIL', 'MTUR', 'REPL',
               'STB', 'SBT', 'TEL', 'TTK', 'TNB', 'UNI', 'VITA', 'OKTA']

    selected_issuer = None
    description = None
    missing_issuer = False

    if request.method == 'POST':
        selected_issuer = request.form.get('issuer')

        if not selected_issuer:
            missing_issuer = True
        else:
            description = get_description_for_issuer(selected_issuer)

    return render_template('fundamentalAnalysis.html', issuers=issuers, selected_issuer=selected_issuer,
                           description=description, missing_issuer=missing_issuer)


# @app.route('/fundamentalAnalysis', methods=['GET', 'POST'])
# def fundamental_analysis():
#     """Render or handle fundamental analysis."""
#     if request.method == 'GET':
#         # Render the form or page for fundamental analysis
#         return render_template('fundamentalAnalysis.html')
#     elif request.method == 'POST':
#         # Communicate with the fundamental analysis service
#         issuer = request.json.get('issuer')
#         response = requests.post('http://fundamental_service:5002/get_description', json={'issuer': issuer})
#         return jsonify(response.json())
#
# @app.route('/lstmPrediction', methods=['GET', 'POST'])
# def lstm_prediction():
#     """Render or handle LSTM prediction."""
#     if request.method == 'GET':
#         # Render the form or page for LSTM prediction
#         return render_template('lstmPrediction.html')
#     elif request.method == 'POST':
#         # Communicate with the LSTM prediction service
#         data = request.json
#         response = requests.post('http://prediction_service:5003/predict', json=data)
#         return jsonify(response.json())


@app.route('/lstmPrediction', methods=['GET', 'POST'])
def lstm_prediction():
    issuers = ['ALK', 'GRNT', 'KMB', 'MPT', 'MTUR', 'OKTA', 'REPL', 'SBT', 'STB',
               'STIL', 'TEL', 'TNB', 'TTK', 'UNI']

    selected_issuer = None
    prediction = None
    missing_issuer = False
    prediction_type = 'short-term'  # Default to 'short-term'

    if request.method == 'POST':
        selected_issuer = request.form.get('issuer')
        prediction_type = request.form.get('prediction_type')  # Get the selected prediction type

        if not selected_issuer:
            missing_issuer = True
        else:
            if prediction_type == 'short-term':
                predictions = get_short_term_predictions('../prediction_service/data/optimized_predictions.csv')
                filtered_predictions = [p for p in predictions if p['Issuer'] == selected_issuer]

                if filtered_predictions:
                    prediction = filtered_predictions[0]
                    prices = [prediction['1 Week'], prediction['1 Month'], prediction['3 Months']]
                    prediction['graph'] = create_graph(selected_issuer, prices, 'short-term')

            elif prediction_type == 'medium-term':
                predictions = get_medium_term_predictions('../prediction_service/data/optimized_predictions_22-24.csv')
                filtered_predictions = [p for p in predictions if p['Issuer'] == selected_issuer]

                if filtered_predictions:
                    prediction = filtered_predictions[0]
                    prices = [prediction['6 Months'], prediction['9 Months'], prediction['1 Year']]
                    prediction['graph'] = create_graph(selected_issuer, prices, 'medium-term')

    return render_template('lstm_prediction.html', issuers=issuers, selected_issuer=selected_issuer,
                           prediction=prediction, missing_issuer=missing_issuer, prediction_type=prediction_type)


@app.route('/about')
def about():
    return render_template('about_us.html')

