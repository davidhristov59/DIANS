import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import logging

# Set up logging for tracking performance
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load and preprocess the data
data = pd.read_csv('../Homework1/filtered_stock_data.csv')

# Data cleaning
data['Last Transaction Price'] = data['Last Transaction Price'].str.replace(r'\.00', '', regex=True)
data['Last Transaction Price'] = data['Last Transaction Price'].str.replace('.', '', regex=False)
data['Last Transaction Price'] = data['Last Transaction Price'].str.replace(',', '.', regex=False)
data['Last Transaction Price'] = pd.to_numeric(data['Last Transaction Price'], errors='coerce')
data = data.dropna(subset=['Last Transaction Price'])

# Parameters
time_steps = 60  # window size
min_data_points = 60  # minimum number of data points to create sequences
min_epoch = 10
max_epoch = 50
batch_size = 32

# Scaling function
def scale_data(Data, feature):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(Data[feature].values.reshape(-1, 1))
    return scaled_data, scaler

# Create sequences for LSTM
def create_sequences(data, time_steps):
    x, y = [], []
    for i in range(time_steps, len(data)):
        x.append(data[i - time_steps:i])
        y.append(data[i])
    return np.array(x), np.array(y)

# Predict multiple future time steps
def predict_multiple_steps(model, last_sequence, steps, scaler):
    prediction_list = []
    current_input = last_sequence

    for _ in range(steps):
        predicted_price = model.predict(current_input)
        prediction_list.append(predicted_price[0][0])
        current_input = np.append(current_input[0][1:], predicted_price, axis=0).reshape(1, -1, 1)

    predictions_rescaled = scaler.inverse_transform(np.array(prediction_list).reshape(-1, 1))
    return predictions_rescaled

# Store all predictions for all issuers
all_predictions = []

# Early stopping for efficient training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Issuer specific training and prediction
for issuer in data['Issuer Name'].unique():
    issuer_data = data[data['Issuer Name'] == issuer]

    if len(issuer_data) >= min_data_points:
        logging.info(f"Processing issuer: {issuer}, Data Points: {len(issuer_data)}")

        # Scale the 'Last Transaction Price'
        scaled_data, scaler = scale_data(issuer_data, 'Last Transaction Price')

        # Create sequences
        X, y = create_sequences(scaled_data, time_steps)

        # Split data into training and validation sets (70% train, 30% validation)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        # Design the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Dynamically adjust epochs based on data size
        epochs = min(max_epoch, max(min_epoch, len(issuer_data) // time_steps))

        # Train the model with early stopping
        logging.info(f"Training LSTM for issuer: {issuer}, Epochs: {epochs}")
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                  validation_data=(X_val, y_val), callbacks=[early_stopping])

        # Predict future stock prices (for 22 trading days as an example)
        last_sequence = X_val[-1].reshape(1, -1, 1)
        predictions = predict_multiple_steps(model, last_sequence, 22, scaler)

        # Calculate average predicted price
        avg_predicted_price = predictions.mean()

        # Log MSE for evaluation
        val_predictions = model.predict(X_val)
        mse = mean_squared_error(y_val, val_predictions)
        logging.info(f'MSE for {issuer}: {mse}')

        # Store the result for this issuer
        all_predictions.append({
            'Issuer': issuer,
            'Average Predicted Price': avg_predicted_price,
            'MSE': mse
        })
    else:
        logging.info(f"Skipping {issuer} due to insufficient data points (< {min_data_points}).")

# Convert all predictions to a DataFrame
predictions_df = pd.DataFrame(all_predictions)

# Save results to file
predictions_df.to_csv('average_predictions_for_all_issuers.csv', index=False)
logging.info("Predictions saved to 'average_predictions_for_all_issuers.csv'.")