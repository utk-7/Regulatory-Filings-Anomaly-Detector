import pandas as pd
import glob
import os

def combine_companies():
    """Combine all company CSV files into one dataframe with ticker column."""

    # Get all CSV files in data directory
    csv_files = glob.glob('data/*.csv')

    print(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")

    combined_data = []
    ticker_summary = {}

    for csv_file in csv_files:
        # Extract ticker from filename (e.g., 'khc_raw.csv' -> 'KHC')
        ticker = os.path.basename(csv_file).replace('_raw.csv', '').upper()

        # Read CSV
        df = pd.read_csv(csv_file)

        # Add ticker column
        df['ticker'] = ticker

        # Calculate quarters of data per ticker
        total_quarters = len(df)
        ticker_summary[ticker] = total_quarters

        # Flag sparse data - a company with less than 5 quarters of data
        is_sparse = total_quarters < 5
        sparse_flag = "YES" if is_sparse else "NO"

        print(f"{ticker}: {total_quarters} quarters of data, sparse data: {sparse_flag}")

        # Add to combined list
        combined_data.append(df)

    # Combine all dataframes
    combined_df = pd.concat(combined_data, ignore_index=True)

    # Reorder columns with ticker first
    cols = ['ticker', 'fiscal_year', 'fiscal_period'] + \
           [col for col in combined_df.columns if col not in ['ticker', 'fiscal_year', 'fiscal_period']]
    combined_df = combined_df[cols]

    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)

    # Save to CSV
    output_path = 'data/all_companies_raw.csv'
    combined_df.to_csv(output_path, index=False)

    print(f"\nCombined data saved to {output_path}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Total unique tickers: {combined_df['ticker'].nunique()}")

    # Print summary
    print("\n=== SUMMARY ===")
    for ticker in sorted(ticker_summary.keys()):
        print(f"{ticker}: {ticker_summary[ticker]} quarters of data")

    return combined_df

if __name__ == "__main__":
    df = combine_companies()