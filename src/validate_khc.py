import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


DATA_CSV = os.path.join('data', 'anomaly_scores.csv')
OUTPUT_PNG = os.path.join('data', 'khc_validation_chart.png')
KRUS = 'KHC'


def main() -> None:
    # 1. Load the anomaly scores
    if not os.path.exists(DATA_CSV):
        print(f"❌ Anomaly scores file not found: {DATA_CSV}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(DATA_CSV)
    # Expect columns: ticker, fiscal_year, fiscal_period, anomaly_score
    required = {"ticker", "fiscal_year", "fiscal_period", "anomaly_score"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(f"❌ Missing columns in {DATA_CSV}: {missing}", file=sys.stderr)
        sys.exit(1)

    df_khc = df[df['ticker'] == KRUS].copy()
    if df_khc.empty:
        print(f"❌ No KHC data found in {DATA_CSV}", file=sys.stderr)
        sys.exit(1)

    # Convert fiscal period (Q1, Q2, …) to a month so we can build a datetime index.
    month_start = {
        'Q1': 1,
        'Q2': 4,
        'Q3': 7,
        'Q4': 10,
    }
    df_khc['month'] = df_khc['fiscal_period'].map(month_start)
    # Build a stable date – first day of each quarter
    df_khc['date'] = pd.to_datetime(dict(year=df_khc['fiscal_year'], month=df_khc['month'], day=1))
    df_khc.sort_values('date', inplace=True)

    # 2. Plot the line chart
    plt.figure(figsize=(10, 5))
    plt.plot(df_khc['date'], df_khc['anomaly_score'], marker='o', label='Anomaly Score')

    # Add vertical marker for Q1 2019 disclosure
    disclosure_date = pd.Timestamp('2019-01-01')
    plt.axvline(disclosure_date, color='red', linestyle='--', linewidth=1.5,
                label='Restatement Disclosure (Q1 2019)')

    plt.title('Kraft Heinz (KHC) Anomaly Score Over Time')
    plt.xlabel('Fiscal Period')
    plt.ylabel('Anomaly Score')
    plt.grid(alpha=0.4)
    plt.legend()

    # 3. Save the figure
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Chart saved to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()