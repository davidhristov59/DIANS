import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time


def fetch_data_for_issuer(issuer_code, start_date, end_date):
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


def fetch_and_filter_missing_data():
    input_file = 'filtered_stock_data.csv'
    try:
        df = pd.read_csv(input_file, parse_dates=['Date'], dayfirst=True)
        print("Existing data loaded successfully.")
    except FileNotFoundError:
        df = pd.DataFrame()
        print("No existing data found. Proceeding to fetch data for the last 12 years.")

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
        "VROS", "VTKS", "ZAS", "ZILU", "ZILUP", "ZIMS", "ZKAR", "ZPKO"
    ]

    all_data = pd.DataFrame()
    today = datetime.now()

    for issuer in issuers:
        if not df.empty and issuer in df['Issuer Name'].values:
            issuer_df = df[df['Issuer Name'] == issuer]
            last_date = issuer_df['Date'].max()
            print(f"Last recorded date for {issuer} is {last_date.strftime('%d.%m.%Y')}")

            if last_date < today - timedelta(days=1):
                start_date = last_date + timedelta(days=1)
                print(
                    f"Fetching missing data for {issuer} from {start_date.strftime('%d.%m.%Y')} to {today.strftime('%d.%m.%Y')}")
                new_data = fetch_data_for_issuer(issuer, start_date, today)
                all_data = pd.concat([all_data, new_data], ignore_index=True)
            else:
                print(f"No additional data needed for {issuer}.")
        else:
            start_date = today - timedelta(days=365 * 12)
            print(
                f"No data found for {issuer}. Fetching data from {start_date.strftime('%d.%m.%Y')} to {today.strftime('%d.%m.%Y')}")
            new_data = fetch_data_for_issuer(issuer, start_date, today)
            all_data = pd.concat([all_data, new_data], ignore_index=True)

    if not all_data.empty:
        filter_todays_stock_data(new_data=all_data)


def filter_todays_stock_data(new_data, output_file='filtered_stock_data.csv'):
    filtered_df = new_data[new_data['Quantity'] != '0']
    if not filtered_df.empty:
        filtered_df.to_csv(output_file, mode='a', index=False, header=not pd.io.common.file_exists(output_file))
        print(f"Filtered new data appended to '{output_file}'.")
    else:
        print("No filtered data to append.")


def main():
    start_time = time.time()  # Start the timer

    fetch_and_filter_missing_data()

    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    print(f"Time taken to execute the code: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
