import pandas as pd


def parse_macedonian_numbers(input_file, output_file):
    df = pd.read_csv(input_file)

    # Columns to parse
    numeric_columns = [
        'Last Transaction Price',
        'Max.',
        'Min.',
        'Average Price'
    ]

    # Parse Macedonian number format in the specified columns
    for col in numeric_columns:
        if col in df.columns:
            # Replace Macedonian format with classical format
            df[col] = (
                df[col]
                .astype(str)  # Ensure all values are strings
                .str.replace(r'\.', '', regex=True)  # Remove thousand separator
                .str.replace(',', '.', regex=True)  # Replace decimal separator
            )
            # Convert to numeric type
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found in the input file.")

    # Save the processed data to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to '{output_file}'.")


def main():
    """
    Main function to parse numbers from Macedonian format in a specific CSV file.
    """
    # input_file = "filtered_stock_data_2024.csv"
    # output_file = "parsedNumbers_data_2024.csv"

    input_file = "filtered_stock_data_2022_2023_2024.csv"
    output_file = "../parsedNumbers_data_2022_24.csv"

    parse_macedonian_numbers(input_file, output_file)


if __name__ == "__main__":
    main()
