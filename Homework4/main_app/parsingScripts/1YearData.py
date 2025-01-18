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


def fetch_last_365_days_data():
    """
    Fetch data for the last 365 days for all issuers.
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

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    print(f"Fetching data from {start_date.strftime('%d.%m.%Y')} to {end_date.strftime('%d.%m.%Y')}")

    for issuer in issuers:
        print(f"Fetching data for {issuer}...")
        new_data = fetch_data_for_issuer(issuer, start_date, end_date)
        all_data = pd.concat([all_data, new_data], ignore_index=True)
        time.sleep(0.5)  # Add a small delay to avoid overwhelming the server

    return all_data


def save_filtered_data(data):
    """
    Filter and save the stock data to a CSV file.
    """
    today = datetime.now().strftime('%Y%m%d')
    output_file = f'stock_data_365days_{today}.csv'

    # Filter out rows where Quantity is 0
    filtered_df = data[data['Quantity'] != '0']
    if not filtered_df.empty:
        filtered_df.to_csv(output_file, index=False)
        print(f"Filtered data saved to '{output_file}'.")
    else:
        print("No valid data to save.")


def main():
    """
    Main function to fetch and save data for the last 365 days.
    """
    start_time = time.time()  # Start the timer

    print("Starting data collection for the last 365 days...")
    all_data = fetch_last_365_days_data()

    # Save the filtered data
    if not all_data.empty:
        save_filtered_data(all_data)

    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    print(f"Time taken to execute the code: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()