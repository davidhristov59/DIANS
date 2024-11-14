import pandas as pd


def check_null_values(file_path):
    try:
        # Load the CSV file
        data = pd.read_csv(file_path)

        # Check for null values
        if data.isnull().values.any():
            print("There are null values in the file.")

            # Optional: Display the columns with null values and count of nulls
            null_counts = data.isnull().sum()
            print("Null values in each column:")
            print(null_counts[null_counts > 0])
        else:
            print("No null values found in the file.")

    except FileNotFoundError:
        print("File not found. Please check the file path and try again.")
    except pd.errors.EmptyDataError:
        print("The file is empty. Please provide a valid CSV file.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
check_null_values('filtered_stock_data.csv')
