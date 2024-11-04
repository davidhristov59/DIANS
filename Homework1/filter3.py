import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import time


def fetch_today_data():
    def fetch_data_for_issuer(issuer_code, startDate, endDate):
        url = 'https://www.mse.mk/mk/stats/symbolhistory/' + issuer_code
        data = {
            'FromDate': startDate.strftime('%d.%m.%Y'),
            'ToDate': endDate.strftime('%d.%m.%Y'),
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

                DF = pd.DataFrame(rows, columns=['Issuer Name'] + headers)
                return DF
            else:
                print(f"No table found for {issuer_code}.")
                return pd.DataFrame()
        else:
            print(f"Failed to retrieve data for {issuer_code}. Status code:", response.status_code)
            return pd.DataFrame()

    now = datetime.now()
    end_date = now  # Set end date to today
    start_date = now  # Set start date to today

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

    start_time = time.time()

    for issuer in issuers:
        df = fetch_data_for_issuer(issuer, start_date, end_date)
        all_data = pd.concat([all_data, df], ignore_index=True)

    if not all_data.empty:
        all_data.to_csv('todays_stock.csv', index=False)
        print("Today's stock data written to 'todays_stock.csv'.")
    else:
        print("No data to save for today.")

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to fetch and save today's stock data: {elapsed_time:.2f} seconds.")


def filter_todays_stock_data(input_file='todays_stock.csv', output_file='filtered_stock_data.csv'):
    df = pd.read_csv(input_file)

    filtered_df = df[df['Quantity'] != 0]

    if not filtered_df.empty:
        filtered_df.to_csv(output_file, mode='a', index=False, header=not pd.io.common.file_exists(output_file))
        print(f"Filtered today's data appended to '{output_file}'.")
    else:
        print("No filtered data to append.")


def main():
    fetch_today_data()
    filter_todays_stock_data()


if __name__ == "__main__":
    main()
