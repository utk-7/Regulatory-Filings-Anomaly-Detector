import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def analyze_khc_position():
    """Analyze KHC position in PCA space, particularly 2018-2019."""

    # Load ratios data
    df = pd.read_csv('data/ratios.csv')
    print(f"Loaded {len(df)} rows from data/ratios.csv")

    # Define the 3 ratio columns
    ratio_cols = ['accruals_ratio', 'receivables_revenue_mismatch', 'debt_equity_trend']

    print(f"\n=== RATIO COMPLETENESS ANALYSIS ===")
    print(f"Rows missing each ratio:")
    for col in ratio_cols:
        missing = df[col].isna().sum()
        print(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")

    # Summary stats for each ratio per ticker
    print(f"\n=== RATIO STATISTICS BY TICKER ===")
    for ticker in sorted(df['ticker'].unique()):
        ticker_data = df[df['ticker'] == ticker]
        print(f"\n{ticker}:")
        for col in ratio_cols:
            available = ticker_data[col].notna().sum()
            if available > 0:
                print(f"  {col}: {available} points, mean={ticker_data[col].mean():.4f}, " +
                      f"std={ticker_data[col].std():.4f}")
            else:
                print(f"  {col}: No data")

    # Create visualization-friendly dataframe
    # For PCA, impute 0 for missing receivables_revenue_mismatch (neutral center value)
    df_viz = df.copy()
    df_viz['receivables_revenue_mismatch'] = df_viz['receivables_revenue_mismatch'].fillna(0)

    # Remove rows where ALL ratios are missing (shouldn't happen but just in case)
    df_viz = df_viz.dropna(subset=['accruals_ratio', 'debt_equity_trend']).copy()

    print(f"\nFinal dataset for PCA: {len(df_viz)} rows")

    # Extract ratio values for PCA
    X = df_viz[ratio_cols].values

    # Apply StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run PCA to 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Add PCA components to dataframe
    df_viz['PC1'] = X_pca[:, 0]
    df_viz['PC2'] = X_pca[:, 1]

    # Print explained variance ratio
    print(f"\n=== PCA EXPLAINED VARIANCE ===")
    print(f"PC1: {pca.explained_variance_ratio_[0]:.4f} ({pca.explained_variance_ratio_[0]*100:.2f}%)")
    print(f"PC2: {pca.explained_variance_ratio_[1]:.4f} ({pca.explained_variance_ratio_[1]*100:.2f}%)")
    print(f"Total: {(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]):.4f} ({(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])*100:.2f}%)")

    # Get KHC data
    khc_data = df_viz[df_viz['ticker'] == 'KHC'].copy()
    other_data = df_viz[df_viz['ticker'] != 'KHC'].copy()

    print(f"\n=== KHC POSITION ANALYSIS ===")
    print(f"KHC data points in PCA: {len(khc_data)}")
    print(f"Other companies data points in PCA: {len(other_data)}")

    if len(khc_data) > 0:
        print(f"\nKHC points in 2018-2019: {len(khc_data[khc_data['fiscal_year'].isin([2018, 2019])])}")

        if len(khc_data[khc_data['fiscal_year'].isin([2018, 2019])]) > 0:
            # Calculate distances from centroid of other companies
            other_centroid = np.array([other_data['PC1'].mean(), other_data['PC2'].mean()])
            print(f"\nOther companies centroid: PC1={other_centroid[0]:.4f}, PC2={other_centroid[1]:.4f}")

            # Analyze KHC 2018-2019
            khc_2018_2019 = khc_data[khc_data['fiscal_year'].isin([2018, 2019])].copy()

            print(f"\nKHC 2018-2019 points:")
            for _, row in khc_2018_2019.iterrows():
                distance = np.sqrt((row['PC1'] - other_centroid[0])**2 + (row['PC2'] - other_centroid[1])**2)
                print(f"  {row['fiscal_year']} Q{row['fiscal_period']}: " +
                      f"PC1={row['PC1']:.4f}, PC2={row['PC2']:.4f}, " +
                      f"accruals={row['accruals_ratio']:.4f}, " +
                      f"mismatch={row['receivables_revenue_mismatch']:.4f}, " +
                      f"debt_trend={row['debt_equity_trend']:.4f}, " +
                      f"Distance from centroid={distance:.4f}")

            # Calculate statistics
            avg_distance = khc_2018_2019['PC1'].apply(
                lambda x: np.sqrt((x - other_centroid[0])**2 + (khc_2018_2019['PC2'].mean() - other_centroid[1])**2)
            ).mean()

            # Calculate typical within-group distances
            other_distances = other_data.apply(
                lambda row: np.sqrt((row['PC1'] - other_centroid[0])**2 + (row['PC2'] - other_centroid[1])**2),
                axis=1
            )
            avg_other_distance = other_distances.mean()

            print(f"\n=== DISTANCE COMPARISON ===")
            print(f"KHC 2018-2019 avg distance from others centroid: {avg_distance:.4f}")
            print(f"Average within-others distance from their centroid: {avg_other_distance:.4f}")

            distance_ratio = avg_distance / avg_other_distance if avg_other_distance > 0 else 0
            print(f"Distance ratio (KHC/others): {distance_ratio:.2f}")

            # Make assessment
            if distance_ratio > 1.5:
                print(f"\nASSESSMENT: KHC 2018-2019 points DRIFT toward the edge of the cluster")
            elif distance_ratio < 0.7:
                print(f"\nASSESSMENT: KHC 2018-2019 points stay MIXED in with other companies")
            else:
                print(f"\nASSESSMENT: KHC 2018-2019 points stay MIXED IN with the other companies (moderate distance)")

    # Also show raw ratio values for 2018-2019 for context
    print(f"\n=== KHC RAW RATIOS (2018-2019) ===")
    khc_raw = df[df['ticker'] == 'KHC']
    khc_2018_2019 = khc_raw[khc_raw['fiscal_year'].isin([2018, 2019])]
    if len(khc_2018_2019) > 0:
        for _, row in khc_2018_2019.iterrows():
            print(f"  {row['fiscal_year']} Q{row['fiscal_period']}: " +
                  f"accruals={row['accruals_ratio']:.4f}, " +
                  f"mismatch={row['receivables_revenue_mismatch'] if pd.notna(row['receivables_revenue_mismatch']) else 'NaN'}, " +
                  f"debt_trend={row['debt_equity_trend'] if pd.notna(row['debt_equity_trend']) else 'NaN'}")

if __name__ == "__main__":
    analyze_khc_position()