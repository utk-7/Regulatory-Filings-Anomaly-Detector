import requests
import pandas as pd
import numpy as np
import json
import os
import re
from datetime import datetime

# SEC API endpoints
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# Placeholder User-Agent - REPLACE WITH YOUR REAL EMAIL
HEADERS = {
    "User-Agent": "SEC Filing Anomaly Detector project user@example.com"
}

def get_cik(ticker):
    """
    Map a stock ticker to its SEC CIK number using the company_tickers.json endpoint.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'KHC')

    Returns:
        str: CIK number as a string, or None if not found
    """
    try:
        response = requests.get(COMPANY_TICKERS_URL, headers=HEADERS)
        response.raise_for_status()
        data = response.json()

        # The data is a dictionary where keys are numeric indices and values are dicts
        # containing 'ticker' and 'cik_str'
        for item in data.values():
            if item['ticker'].upper() == ticker.upper():
                return item['cik_str']

        print(f"Warning: Ticker {ticker} not found in company_tickers.json")
        return None
    except Exception as e:
        print(f"Error fetching CIK for ticker {ticker}: {e}")
        return None

def get_company_facts(cik):
    """
    Fetch company facts for a given CIK from the SEC EDGAR API.

    Args:
        cik (str): CIK number (will be zero-padded to 10 digits in the URL)

    Returns:
        dict: JSON response containing company facts, or None if failed
    """
    try:
        # Zero-pad CIK to 10 digits as required by the URL format
        cik_padded = str(cik).zfill(10)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"

        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching company facts for CIK {cik}: {e}")
        return None

def extract_quarterly_data(facts_data, tags_to_extract):
    """
    Extract quarterly financial data from company facts for specified tags.

    Args:
        facts_data (dict): Company facts JSON from get_company_facts()
        tags_to_extract (list): List of US-GAAP tags to extract (with fallbacks)

    Returns:
        pd.DataFrame: Dataframe with one row per fiscal quarter containing the requested metrics
    """
    if not facts_data or 'facts' not in facts_data or 'us-gaap' not in facts_data.get('facts', {}):
        print("Error: Invalid or missing us-gaap data in company facts")
        return pd.DataFrame()

    us_gaap_data = facts_data['facts']['us-gaap']

    # We'll store data by quarter (fiscal year, fiscal period)
    quarterly_data = {}

    # Process each tag
    for tag_info in tags_to_extract:
        # Handle fallback tags
        primary_tag = tag_info['primary']
        fallback_tag = tag_info.get('fallback')

        # Try primary tag first, then fallback if specified
        tag_data = None
        used_tag = None

        if primary_tag in us_gaap_data:
            tag_data = us_gaap_data[primary_tag]
            used_tag = primary_tag
        elif fallback_tag and fallback_tag in us_gaap_data:
            tag_data = us_gaap_data[fallback_tag]
            used_tag = fallback_tag
            print(f"Note: Using fallback tag '{fallback_tag}' for '{primary_tag}'")
        else:
            print(f"Warning: Tag '{primary_tag}' not found in company facts" +
                  (f" (and fallback '{fallback_tag}' also not found)" if fallback_tag else ""))
            continue

        # Extract USD units data
        if 'units' not in tag_data or 'USD' not in tag_data['units']:
            print(f"Warning: No USD units found for tag '{used_tag}'")
            continue

        entries = tag_data['units']['USD']

        # Process each entry to get the latest value for each quarter
        quarter_values = {}  # Key: (fy, fp), Value: (value, filed_date)

        for entry in entries:
            # Extract relevant fields
            fy = entry.get('fy')
            fp = entry.get('fp')
            val = entry.get('val')
            end_date = entry.get('end')
            filed_date = entry.get('filed')

            # Skip if no value
            if val is None:
                continue

            # Try to get quarter info - prefer fp, fallback to parsing frame
            if fp is None or fp not in ['Q1', 'Q2', 'Q3', 'Q4']:
                # Parse from 'frame' field (e.g., 'CY2013Q4I')
                frame = entry.get('frame', '')
                match = re.search(r'CY(\d{4})Q([1-4])', frame)
                if match:
                    fy = int(match.group(1))
                    fp = f"Q{match.group(2)}"
                else:
                    continue

            if fy is None or fp is None:
                continue

            # Convert fy to int if it's still a string
            try:
                fy = int(fy)
            except (ValueError, TypeError):
                continue

            # Create quarter key
            quarter_key = (fy, fp)

            # If we already have a value for this quarter, keep the one with the latest filing date
            if quarter_key in quarter_values:
                existing_filed = quarter_values[quarter_key][1]
                # If current entry has a later filed date, use it
                if filed_date and (not existing_filed or filed_date > existing_filed):
                    quarter_values[quarter_key] = (val, filed_date)
            else:
                quarter_values[quarter_key] = (val, filed_date)

        # Store the quarterly values for this tag
        for (fy, fp), (value, _) in quarter_values.items():
            if (fy, fp) not in quarterly_data:
                quarterly_data[(fy, fp)] = {}
            quarterly_data[(fy, fp)][used_tag] = value

    # Convert to DataFrame
    if not quarterly_data:
        print("Warning: No quarterly data extracted")
        return pd.DataFrame()

    # Create DataFrame from the quarterly_data dictionary
    df_rows = []
    for (fy, fp), values in quarterly_data.items():
        row = {'fiscal_year': fy, 'fiscal_period': fp}
        row.update(values)
        df_rows.append(row)

    df = pd.DataFrame(df_rows)

    # Sort by fiscal year and then fiscal period (Q1, Q2, Q3, Q4)
    df['fiscal_period'] = pd.Categorical(df['fiscal_period'],
                                         categories=['Q1', 'Q2', 'Q3', 'Q4'],
                                         ordered=True)
    df = df.sort_values(['fiscal_year', 'fiscal_period']).reset_index(drop=True)

    # Reorder columns to have fiscal_year and fiscal_period first
    cols = ['fiscal_year', 'fiscal_period'] + [col for col in df.columns if col not in ['fiscal_year', 'fiscal_period']]
    df = df[cols]

    return df

def main():
    """Main function to fetch KHC data and save to CSV."""
    print("Fetching SEC financial data for Kraft Heinz (KHC)...")

    # Get CIK for KHC
    cik = get_cik('KHC')
    if not cik:
        print("Failed to get CIK for KHC. Exiting.")
        return

    print(f"Found CIK for KHC: {cik}")

    # Get company facts
    facts_data = get_company_facts(cik)
    if not facts_data:
        print("Failed to get company facts. Exiting.")
        return

    print("Successfully retrieved company facts")

    # Define tags to extract with fallbacks where specified
    tags_to_extract = [
        {'primary': 'Revenues', 'fallback': 'RevenueFromContractWithCustomerExcludingAssessedTax'},
        {'primary': 'NetIncomeLoss'},
        {'primary': 'Assets'},
        {'primary': 'Liabilities'},
        {'primary': 'AccountsReceivableNetCurrent'},
        {'primary': 'NetCashProvidedByUsedInOperatingActivities'}
    ]

    # Extract quarterly data
    df = extract_quarterly_data(facts_data, tags_to_extract)

    if df.empty:
        print("No data extracted. Exiting.")
        return

    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)

    # Save to CSV
    output_path = 'data/khc_raw.csv'
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

    # Print first 5 rows
    print("\nFirst 5 rows of the extracted data:")
    print(df.head())

    print(f"\nTotal rows extracted: {len(df)}")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    main()