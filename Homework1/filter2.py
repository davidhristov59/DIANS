import pandas as pd
from datetime import datetime, timedelta

# File paths
input_file = '../Homework3/filtered_stock_data.csv'

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
    "SPAZ", "SPAZP", "STB", "STBP", "STIL", "STOK", "TAJM"
    , "TEAL", "TEHN", "TEL", "TETE", "TIKV", "TKPR", "TKVS", "TNB",
    "TRDB", "TRPS",
    "TSMP", "TTK", "UNI", "USJE", "VITA",
    "VROS", "VTKS", "ZAS", "ZILU", "ZILUP", "ZIMS", "ZKAR", "ZPKO"
]

for issuer in issuers:
    if not df.empty and issuer in df['Issuer Name'].values:
        # Filter data for the specific issuer
        issuer_df = df[df['Issuer Name'] == issuer]

        last_date = issuer_df['Date'].max()
        print(f"Last recorded date for {issuer} is {last_date.strftime('%d.%m.%Y')}")

        if last_date < datetime.now() - timedelta(days=365 * 10):
            print(f"Fetching data for {issuer} for the last 10 years from {last_date + timedelta(days=1)}.")

        else:
            print(f"No additional data needed for {issuer}.")
    else:
        start_date = datetime.now() - timedelta(days=365 * 12)
        print(f"No data found for {issuer}. Fetching data from {start_date.strftime('%d.%m.%Y')}.")
