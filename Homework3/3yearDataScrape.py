import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time


def fetch_data_for_issuer(issuer_code, start_date, end_date):
    """
    Fetch historical data for a specific issuer from the MSE website.
    """
    url = 'https://www.mse.mk/mk/stats/symbolhistory/' + issuer_code
    data = {
        'FromDate': start_date.strftime('%d.%m.%Y'),
        'ToDate': end_date.strftime('%d.%m.%Y'),
        'Code': issuer_code
    }

    response = requests.post(url, data=data)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        headers = [
            'Date',
            'Last Transaction Price',
            'Max.',
            'Min.',
            'Average Price',
            '% Change',
            'Quantity',
            'Turnover in BEST in Denars',
            'Total Turnover in Denars'
        ]
        table = soup.find('table')

        if table:
            rows = []
            for row in table.find_all('tr'):
                cells = row.find_all('td')
                if cells:
                    row_data = [cell.get_text(strip=True) for cell in cells]
                    row_data.insert(0, issuer_code)  # Add issuer code as the first column
                    rows.append(row_data)

            df = pd.DataFrame(rows, columns=['Issuer Name'] + headers)
            return df
        else:
            print(f"No table found for {issuer_code}.")
            return pd.DataFrame()
    else:
        print(f"Failed to retrieve data for {issuer_code}. Status code:", response.status_code)
        return pd.DataFrame()


def fetch_data_for_2022_2023_2024():
    """
    Fetch data for the years 2022, 2023, and 2024 for all issuers.
    """
    issuers = [
        "ADIN", "ALK", "CKB", "DEBA", "DIMI",
        "EUHA", "EVRO", "FAKM", "FERS",
        "FUBT", "GALE", "GECK", "GECT", "GIMS", "GRNT", "GRZD",
        "GTC", "GTRG", "INB", "KARO",
        "KDFO", "KJUBI", "KLST", "KMB", "KOMU", "KONF", "KONZ",
        "KPSS", "KVAS", "LOTO", "LOZP", "MAKP", "MAKS",
        "MB", "MERM", "MKSD", "MPOL", "MPT", "MTUR",
        "MZPU", "NEME", "NOSK", "OKTA",
        "OTEK", "PKB", "POPK", "PPIV", "PROD", "RADE",
        "REPL", "RZTK", "RZUG",
        "RZUS", "SBT", "SDOM", "SIL", "SKP", "SLAV", "SOLN",
        "SPAZ", "SPAZP", "STB", "STBP", "STIL", "STOK", "TAJM",
        "TEAL", "TEHN", "TEL", "TETE", "TIKV", "TKPR", "TKVS", "TNB",
        "TRDB", "TRPS",
        "TSMP", "TTK", "UNI", "USJE", "VITA",
        "VROS", "VTKS", "ZAS", "ZILU", "ZILUP", "ZIMS", "ZKAR", "ZPKO", "OMOS"
    ]

    all_data = pd.DataFrame()

    # Define specific date ranges for the years 2022, 2023, and 2024
    # From 01.01.2022 to 31.12.2022
    start_date_2022 = datetime(2022, 1, 1)
    end_date_2022 = datetime(2022, 12, 31)

    # From 01.01.2023 to 31.12.2023
    start_date_2023 = datetime(2023, 1, 1)
    end_date_2023 = datetime(2023, 12, 31)

    # From 01.01.2024 to today's date
    start_date_2024 = datetime(2024, 1, 1)
    end_date_2024 = datetime.now()

    # Fetch data for 2022, 2023, and 2024
    for issuer in issuers:
        # Fetch data for each year
        print(f"Fetching data for {issuer} for the year 2022...")
        new_data_2022 = fetch_data_for_issuer(issuer, start_date_2022, end_date_2022)
        all_data = pd.concat([all_data, new_data_2022], ignore_index=True)

        print(f"Fetching data for {issuer} for the year 2023...")
        new_data_2023 = fetch_data_for_issuer(issuer, start_date_2023, end_date_2023)
        all_data = pd.concat([all_data, new_data_2023], ignore_index=True)

        print(f"Fetching data for {issuer} for the year 2024...")
        new_data_2024 = fetch_data_for_issuer(issuer, start_date_2024, end_date_2024)
        all_data = pd.concat([all_data, new_data_2024], ignore_index=True)

    return all_data


def save_filtered_data(data, output_file='filtered_stock_data_2022_2023_2024.csv'):
    """
    Filter and save the stock data to a CSV file.
    """
    # Filter out rows where Quantity is 0
    filtered_df = data[data['Quantity'] != '0']
    if not filtered_df.empty:
        filtered_df.to_csv(output_file, index=False)
        print(f"Filtered data saved to '{output_file}'.")
    else:
        print("No valid data to save.")


def main():
    """
    Main function to fetch and save data for 2022, 2023, and 2024.
    """
    start_time = time.time()  # Start the timer

    # Fetch data for 2022, 2023, and 2024
    all_data = fetch_data_for_2022_2023_2024()

    # Save the filtered data
    if not all_data.empty:
        save_filtered_data(all_data)

    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    print(f"Time taken to execute the code: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
