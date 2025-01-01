from flask import Flask
from controller import index, get_todays_data_route, technical_analysis, fundamental_analysis, lstm_prediction, about

app = Flask(__name__)

# Register routes
app.add_url_rule('/', 'index', index)
app.add_url_rule('/get_todays_data', 'get_todays_data', get_todays_data_route, methods=['GET'])
app.add_url_rule('/technicalAnalysis', 'technical_analysis', technical_analysis, methods=['GET', 'POST'])
app.add_url_rule('/fundamentalAnalysis', 'fundamental_analysis', fundamental_analysis, methods=['GET', 'POST'])
app.add_url_rule('/lstmPrediction', 'lstm_prediction', lstm_prediction, methods=['GET', 'POST'])
app.add_url_rule('/about', 'about', about)

if __name__ == '__main__':
    app.run(debug=True)
