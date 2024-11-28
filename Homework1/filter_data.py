import pandas as pd

# Read the CSV file
input_file = 'historical_stock_data.csv'
output_file = '../Homework3/filtered_stock_data.csv'

df = pd.read_csv(input_file)

# Filter out rows where the 'Quantity' column is 0
filtered_df = df[df['Quantity'] != 0]

filtered_df.to_csv(output_file, index=False)

print(f"Filtered data successfully written to '{output_file}'.")
