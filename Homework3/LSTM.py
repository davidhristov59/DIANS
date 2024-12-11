import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Load the stock data
data = pd.read_csv('../Homework1/filtered_stock_data.csv')

# Remove trailing ',00' and replace thousand separators and decimal commas
data['Last Transaction Price'] = data['Last Transaction Price'].str.replace(r'\.00$', '', regex=True)  # Remove ',00'
data['Last Transaction Price'] = data['Last Transaction Price'].str.replace('.', '', regex=False)  # Remove a thousand separator
data['Last Transaction Price'] = data['Last Transaction Price'].str.replace(',', '.', regex=False)  # Convert decimal comma to period

# Convert the 'Last Transaction Price' column to numeric
data['Last Transaction Price'] = pd.to_numeric(data['Last Transaction Price'], errors='coerce')

# Drop rows with NaN values in 'Last Transaction Price'
data = data.dropna(subset=['Last Transaction Price'])

# Parameters
time_steps = 60  # window size
min_data_points = 60  # minimum number of data points to create sequences
min_epoch = 1
max_epoch = 50

# Scaling function
def scale_data(data, feature):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[feature].values.reshape(-1, 1))
    return scaled_data, scaler

# Function to create sequences for LSTM
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Function to predict for multiple future time steps
def predict_multiple_steps(model, last_sequence, steps, scaler):
    prediction_list = []
    current_input = last_sequence

    for _ in range(steps):
        predicted_price = model.predict(current_input)
        prediction_list.append(predicted_price[0][0])

        # Prepare the next input by appending the predicted price
        current_input = np.append(current_input[0][1:], predicted_price, axis=0).reshape(1, -1, 1)

    # Rescale back to original price scale
    predictions_rescaled = scaler.inverse_transform(np.array(prediction_list).reshape(-1, 1))
    return predictions_rescaled

# Store all predictions for all issuers in a list
all_predictions = []

# Issuer specific training and prediction
for issuer in data['Issuer Name'].unique():
    issuer_data = data[data['Issuer Name'] == issuer]

    # Only proceed if issuer has enough data points
    if len(issuer_data) >= min_data_points:
        print(f"Processing issuer: {issuer}, Data Points: {len(issuer_data)}")

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
        epochs = min(max_epoch, max(min_epoch, len(issuer_data) // 60))

        # Train the model
        print(f"Training LSTM for issuer: {issuer}, Epochs: {epochs}")
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val))

        # Predict future stock prices
        last_sequence = X_val[-1].reshape(1, -1, 1)

        # Predict for 1 Month (22 trading days, as an example)
        predictions = predict_multiple_steps(model, last_sequence, 22, scaler)

        # Calculate the average predicted price
        avg_predicted_price = predictions.mean()

        # Store the result for this issuer
        all_predictions.append({
            'Issuer': issuer,
            'Average Predicted Price': avg_predicted_price
        })

    else:
        print(f"Skipping {issuer} as it has less than {min_data_points} data points.")

# Convert the list of all predictions to a DataFrame
predictions_df = pd.DataFrame(all_predictions)

# Save the predictions to a TXT file
with open('average_predictions_for_all_issuers.txt', 'w') as file:
    for index, row in predictions_df.iterrows():
        file.write(f"Issuer: {row['Issuer']}, Average Predicted Price: {row['Average Predicted Price']}\n")

print("All average predictions have been saved to 'average_predictions_for_all_issuers.txt'.")