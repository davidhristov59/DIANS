from flask import Flask, request, jsonify, render_template
from controller import index, get_todays_data_route, technical_analysis, fundamental_analysis,lstm_prediction
import requests

app = Flask(__name__)

app.add_url_rule('/', 'index', index)
app.add_url_rule('/get_todays_data', 'get_todays_data', get_todays_data_route, methods=['GET'])
app.add_url_rule('/technicalAnalysis', 'technical_analysis', technical_analysis, methods=['GET', 'POST'])
app.add_url_rule('/fundamentalAnalysis', 'fundamental_analysis', fundamental_analysis, methods=['GET', 'POST'])

app.add_url_rule('/lstmPrediction', 'lstm_prediction', lstm_prediction, methods=['GET', 'POST'])


# Use environment variable with fallback
PREDICTION_SERVICE_URL = 'http://prediction-service:5003/predict'



@app.route('/predict', methods=['GET', 'POST'])
def lstm_prediction1():
    issuers = ['ALK', 'GRNT', 'KMB', 'MPT', 'MTUR', 'OKTA', 'REPL', 'SBT', 'STB',
               'STIL', 'TEL', 'TNB', 'TTK', 'UNI']

    selected_issuer = None
    prediction = None
    missing_issuer = False
    prediction_type = 'short-term'  # Default to 'short-term'

    if request.method == 'POST':
        try:
            selected_issuer = request.form.get('issuer')
            prediction_type = request.form.get('prediction_type', 'short-term')

            if not selected_issuer:
                missing_issuer = True
            else:
                print(f"Attempting to connect to: {PREDICTION_SERVICE_URL}")
                data = {
                    'issuer': selected_issuer,
                    'prediction_type': prediction_type
                }
                print(f"Sending data: {data}")

                # Add timeout and more detailed error handling
                try:
                    response = requests.post(
                        PREDICTION_SERVICE_URL,
                        json=data,
                        timeout=5  # 5 seconds timeout
                    )
                    print(f"Response status: {response.status_code}")
                    print(f"Response content: {response.content}")

                    if response.status_code == 200:
                        prediction = response.json()
                    else:
                        print(f"Error response: {response.text}")
                        return jsonify({
                            'error': 'Error from prediction service',
                            'status_code': response.status_code,
                            'details': response.text
                        }), response.status_code
                except requests.exceptions.Timeout:
                    print("Request timed out")
                    return jsonify({'error': 'Request to prediction service timed out'}), 504
                except requests.exceptions.ConnectionError as e:
                    print(f"Connection error: {str(e)}")
                    return jsonify({'error': 'Could not connect to prediction service', 'details': str(e)}), 503
                except Exception as e:
                    print(f"Unexpected error: {str(e)}")
                    return jsonify({'error': 'Unexpected error', 'details': str(e)}), 500
        except Exception as e:
            print(f"Error in lstm_prediction1: {str(e)}")
            return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

    return render_template('lstm_prediction.html',
                           issuers=issuers,
                           selected_issuer=selected_issuer,
                           prediction=prediction,
                           missing_issuer=missing_issuer,
                           prediction_type=prediction_type)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
