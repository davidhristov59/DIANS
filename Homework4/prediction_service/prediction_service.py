from flask import Flask, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Function to get short-term predictions from the CSV
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

# Function to get medium-term predictions from the CSV
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

# Function to create a graph of predictions
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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    issuer = data.get('issuer')
    prediction_type = data.get('prediction_type', 'short-term')

    if not issuer:
        return jsonify({'error': 'Issuer is required'}), 400

    # Select the right prediction file based on prediction type
    if prediction_type == 'short-term':
        predictions = get_short_term_predictions('data/optimized_predictions.csv')
    elif prediction_type == 'medium-term':
        predictions = get_medium_term_predictions('data/optimized_predictions_22-24.csv')
    else:
        return jsonify({'error': 'Invalid prediction type'}), 400

    # Filter predictions for the selected issuer
    filtered_predictions = [p for p in predictions if p['Issuer'] == issuer]
    if not filtered_predictions:
        return jsonify({'error': 'Issuer not found'}), 404

    prediction = filtered_predictions[0]
    if prediction_type == 'short-term':
        prices = [prediction['1 Week'], prediction['1 Month'], prediction['3 Months']]
    else:
        prices = [prediction['6 Months'], prediction['9 Months'], prediction['1 Year']]

    # Create the graph and add it to the prediction
    prediction['graph'] = create_graph(issuer, prices, prediction_type)

    return jsonify(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
