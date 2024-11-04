import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta


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


end_date = datetime.now()
# 12 years of data
start_date = end_date - timedelta(days=365 * 12)
# start_date = end_date - timedelta(days=3650)  # 12 years ago

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
    "SPAZ", "SPAZP", "STB", "STBP", "STIL", "STOK", "TAJM"
    , "TEAL", "TEHN", "TEL", "TETE", "TIKV", "TKPR", "TKVS", "TNB",
    "TRDB", "TRPS",
    "TSMP", "TTK", "UNI", "USJE", "VITA",
    "VROS", "VTKS", "ZAS", "ZILU", "ZILUP", "ZIMS", "ZKAR", "ZPKO"
]

all_data = pd.DataFrame()

# Fetch data for each issuer for each year
for year in range(12):
    current_start_date = start_date + timedelta(days=365 * year)
    current_end_date = current_start_date + timedelta(days=365)

    for issuer in issuers:
        df = fetch_data_for_issuer(issuer, current_start_date, current_end_date)
        all_data = pd.concat([all_data, df], ignore_index=True)

all_data.to_csv('historical_stock_data.csv', index=False)
print("Historical stock data successfully written to 'historical_stock_data.csv'.")
