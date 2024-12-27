import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Load and preprocess data
# data = pd.read_csv("parsingScripts/parsedNumbers_data_2024.csv")
data = pd.read_csv("parsingScripts/parsedNumbers_data_2022_24.csv")

# Parameters
TIME_STEPS = 60
BATCH_SIZE = 32
MIN_EPOCHS = 10
MAX_EPOCHS = 50
MIN_DATA_POINTS = 60


# Helper functions
def scale_data(data, feature):
    """Scale data using MinMaxScaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[feature].values.reshape(-1, 1))
    return scaled_data, scaler


def create_sequences(data, time_steps):
    """Generate sequences and labels for LSTM training."""
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def predict_future(model, last_sequence, steps, scaler):
    """Predict future values using the trained model."""
    predictions = []
    current_input = last_sequence
    for _ in range(steps):
        pred = model.predict(current_input)
        predictions.append(pred[0][0])
        current_input = np.append(current_input[0][1:], pred, axis=0).reshape(1, -1, 1)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))


# Initialize callbacks
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)

# Processing for each issuer
time_horizons = {
    # "1 Week": 5,
    # "1 Month": 22,
    # "3 Months": 66

    "6 Months": 132,
    "9 Months": 198,
    "1 Year": 264
}
all_predictions = []

for issuer in data["Issuer Name"].unique():
    issuer_data = data[data["Issuer Name"] == issuer].copy()

    # Ensure data is sorted by the latest available date
    issuer_data.sort_values(by="Date", ascending=True, inplace=True)

    if len(issuer_data) < MIN_DATA_POINTS:
        logging.warning(f"Skipping issuer {issuer}: insufficient data points ({len(issuer_data)}).")
        continue

    logging.info(f"Processing issuer: {issuer} with {len(issuer_data)} data points.")

    # Scale the feature
    scaled_data, scaler = scale_data(issuer_data, "Last Transaction Price")

    # Create sequences
    X, y = create_sequences(scaled_data, TIME_STEPS)

    # Split data into training and validation sets
    split_index = int(0.8 * len(X))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    # Design LSTM model
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(units=64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(units=1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train the model
    epochs = min(MAX_EPOCHS, max(MIN_EPOCHS, len(issuer_data) // TIME_STEPS))
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, lr_scheduler]
    )

    # Predict future time horizons
    last_sequence = X_val[-1].reshape(1, -1, 1)
    predictions = {"Issuer": issuer}

    for horizon, steps in time_horizons.items():
        pred_values = predict_future(model, last_sequence, steps, scaler)
        predictions[horizon] = pred_values.mean()

    # Validate and log metrics
    val_preds = model.predict(X_val)
    mse = mean_squared_error(y_val, val_preds)
    mae = mean_absolute_error(y_val, val_preds)
    predictions["MSE"] = mse
    predictions["MAE"] = mae

    logging.info(f"Issuer {issuer} - MSE: {mse}, MAE: {mae}")
    all_predictions.append(predictions)

# Save predictions to CSV
predictions_df = pd.DataFrame(all_predictions)
# predictions_df.to_csv("optimized_predictions.csv", index=False)
# logging.info("Predictions saved to 'optimized_predictions.csv'.")
predictions_df.to_csv("optimized_predictions_22-24.csv", index=False)
logging.info("Predictions saved to 'optimized_predictions_22-24.csv'.")
