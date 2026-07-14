import pandas as pd
import numpy as np
import os

# The 6 raw fields needed across all 3 ratios
REQUIRED_FIELDS = [
    'NetIncomeLoss',
    'NetCashProvidedByUsedInOperatingActivities',
    'Assets',
    'Revenues',
    'AccountsReceivableNetCurrent',
    'Liabilities'
]

def compute_ratios():
    """Calculate the 3 anomaly ratios per company-quarter."""

    # Read combined data
    input_path = 'data/all_companies_raw.csv'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run combine_companies.py first.")
        return None

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Tickers: {sorted(df['ticker'].unique())}")

    # --- Step 1: Drop rows missing any of the 6 required raw fields ---
    initial_count = len(df)

    # Count missing per field for reporting
    missing_counts = {}
    for field in REQUIRED_FIELDS:
        missing_counts[field] = int(df[field].isna().sum())

    # Drop rows where ANY required field is missing
    df_clean = df.dropna(subset=REQUIRED_FIELDS).copy()

    dropped_count = initial_count - len(df_clean)
    print(f"\n=== MISSING VALUE HANDLING ===")
    print(f"Initial rows: {initial_count}")
    print(f"Rows dropped (missing any required raw field): {dropped_count}")
    print(f"Rows remaining: {len(df_clean)}")
    print("\nMissing values per field (in initial data):")
    for field, count in missing_counts.items():
        if count > 0:
            print(f"  {field}: {count} missing")
        else:
            print(f"  {field}: 0 missing (complete)")

    # --- Step 2: Order for QoQ computation ---
    # Ensure fiscal_period is ordered Q1 < Q2 < Q3 < Q4
    df_clean['fiscal_period'] = pd.Categorical(
        df_clean['fiscal_period'],
        categories=['Q1', 'Q2', 'Q3', 'Q4'],
        ordered=True
    )
    df_clean = df_clean.sort_values(
        ['ticker', 'fiscal_year', 'fiscal_period']
    ).reset_index(drop=True)

    # --- Step 3: QoQ helper columns within each ticker ---
    # QoQ % change in AccountsReceivableNetCurrent
    df_clean['AR_qoq_pct'] = df_clean.groupby('ticker')['AccountsReceivableNetCurrent'].pct_change()
    # QoQ % change in Revenues
    df_clean['Rev_qoq_pct'] = df_clean.groupby('ticker')['Revenues'].pct_change()
    # QoQ change in [Liabilities / (Assets - Liabilities)]
    df_clean['debt_equity_ratio'] = df_clean['Liabilities'] / (df_clean['Assets'] - df_clean['Liabilities'])
    df_clean['debt_equity_trend'] = df_clean.groupby('ticker')['debt_equity_ratio'].diff()

    # --- Step 4: Compute the 3 ratios ---
    # 1. Accruals ratio
    df_clean['accruals_ratio'] = (
        df_clean['NetIncomeLoss'] - df_clean['NetCashProvidedByUsedInOperatingActivities']
    ) / df_clean['Assets']

    # 2. Receivables-revenue mismatch
    df_clean['receivables_revenue_mismatch'] = (
        df_clean['AR_qoq_pct'] - df_clean['Rev_qoq_pct']
    )

    # 3. Debt-equity trend (already QoQ change in the ratio)
    # df_clean['debt_equity_trend'] computed above

    # --- Step 5: Build output dataframe ---
    output_cols = [
        'ticker', 'fiscal_year', 'fiscal_period',
        'accruals_ratio', 'receivables_revenue_mismatch', 'debt_equity_trend'
    ]
    result = df_clean[output_cols].copy()

    # Ensure output directory exists
    os.makedirs('data', exist_ok=True)

    # Save to CSV
    output_path = 'data/ratios.csv'
    result.to_csv(output_path, index=False)
    print(f"\nRatios saved to {output_path}")
    print(f"Total rows in output: {len(result)}")

    return result

if __name__ == "__main__":
    result = compute_ratios()
    if result is not None:
        print("\n=== FIRST 10 ROWS ===")
        print(result.head(10).to_string())

        print("\n=== SUMMARY STATISTICS (3 ratio columns) ===")
        print(result[['accruals_ratio', 'receivables_revenue_mismatch', 'debt_equity_trend']].describe().to_string())
