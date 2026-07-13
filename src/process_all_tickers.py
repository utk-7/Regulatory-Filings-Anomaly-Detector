#!/usr/bin/env python3

import pandas as pd
import numpy as np
from fetch_edgar import get_cik, get_company_facts, extract_quarterly_data

TICKERS = ['KHC', 'GIS', 'CPB', 'CAG', 'MDLZ', 'HRL', 'K', 'HSY', 'SJM']
TAGS_TO_EXTRACT = [
    {'primary': 'Revenues', 'fallback': 'RevenueFromContractWithCustomerExcludingAssessedTax'},
    {'primary': 'NetIncomeLoss'},
    {'primary': 'Assets'},
    {'primary': 'Liabilities'},
    {'primary': 'AccountsReceivableNetCurrent'},
    {'primary': 'NetCashProvidedByUsedInOperatingActivities'}
]

all_data = []

for ticker in TICKERS:
    print(f'Processing {ticker}...')
    try:
        cik = get_cik(ticker)
        if not cik:
            print(f'Skipping {ticker} - no CIK found')
            continue

        facts_data = get_company_facts(cik)
        if not facts_data:
            print(f'Skipping {ticker} - failed to get facts')
            continue

        df = extract_quarterly_data(facts_data, TAGS_TO_EXTRACT)
        if df.empty:
            print(f'Skipping {ticker} - no data extracted')
            continue

        df['ticker'] = ticker
        all_data.append(df)

    except Exception as e:
        print(f'Error processing {ticker}: {str(e)}')

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv('data/all_companies_raw.csv', index=False)
    print(f'\nSaved {len(all_data)} company datasets to data/all_companies_raw.csv')
    print(f'Total rows: {len(combined_df)}')
    print(f'Unique tickers: {len(combined_df['ticker'].unique())}')
    print(f'\nFirst 5 rows:')
    print(combined_df.head(5))
else:
    print('\nNo data was collected. Check ticker validity or API access.')