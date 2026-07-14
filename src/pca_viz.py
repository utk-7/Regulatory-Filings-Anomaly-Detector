import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

def pca_visualization():
    """Load ratios data, apply PCA, and create scatter plot."""

    # Load ratios data
    input_path = 'data/ratios.csv'
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    # Define the 3 ratio columns
    ratio_cols = ['accruals_ratio', 'receivables_revenue_mismatch', 'debt_equity_trend']

    # Check which tickers are missing which ratios
    print("\n=== DATA COMPLETENESS BY TICKER ===")
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker]
        missing_cols = ticker_df[ratio_cols].isna().sum()
        print(f"{ticker}: accruals={missing_cols['accruals_ratio']}, " +
              f"receivables_mismatch={missing_cols['receivables_revenue_mismatch']}, " +
              f"debt_equity={missing_cols['debt_equity_trend']}")

    # For visualization, we'll use 2D PCA on available ratios
    # For tickers with complete data: use all 3 ratios
    # For tickers with partial data: use only available ratios (impute 0 for missing)

    # Create a visualization-friendly dataframe
    df_viz = df.copy()

    # For tickers with missing receivables_revenue_mismatch, we need to handle it
    # We'll use a 2-component PCA based on available data per ticker
    # For simplicity, impute 0 for missing receivables_revenue_mismatch (neutral value)
    # This allows all tickers to be visualized in the same 3D->2D space

    # Impute 0 for missing receivables_revenue_mismatch (neutral center value)
    df_viz['receivables_revenue_mismatch'] = df_viz['receivables_revenue_mismatch'].fillna(0)

    # Now drop rows where accruals_ratio or debt_equity_trend are missing (core ratios)
    df_viz = df_viz.dropna(subset=['accruals_ratio', 'debt_equity_trend']).copy()
    dropped_for_pca = len(df) - len(df_viz)
    print(f"\nRows dropped due to missing core ratios: {dropped_for_pca}")

    if len(df_viz) == 0:
        print("Error: No rows available for PCA")
        return None

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

    # Create scatter plot
    plt.figure(figsize=(12, 8))

    # Get unique tickers
    tickers = df_viz['ticker'].unique()

    # Define colors for each ticker
    colors = plt.cm.tab10(np.linspace(0, 1, len(tickers)))
    color_map = dict(zip(tickers, colors))

    # Track which tickers have KHC data
    khc_in_data = 'KHC' in tickers
    other_tickers = [t for t in tickers if t != 'KHC']

    # Plot KHC first (if present) with special highlighting
    if khc_in_data:
        khc_data = df_viz[df_viz['ticker'] == 'KHC']
        plt.scatter(khc_data['PC1'], khc_data['PC2'],
                   s=150,  # larger marker
                   c=[color_map['KHC']],
                   edgecolors='black',
                   linewidths=2,
                   marker='o',
                   label='KHC (highlighted)',
                   zorder=5)

    # Plot other tickers
    for ticker in other_tickers:
        ticker_data = df_viz[df_viz['ticker'] == ticker]
        plt.scatter(ticker_data['PC1'], ticker_data['PC2'],
                   s=60,  # smaller marker
                   c=[color_map[ticker]],
                   edgecolors='white',
                   marker='o',
                   label=ticker,
                   alpha=0.7,
                   zorder=3)

    # Add labels, title, legend
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.title('PCA of Financial Ratios by Company-Quarter\n(KHC highlighted with larger markers)', fontsize=14)

    # Create legend
    legend_elements = []
    if khc_in_data:
        legend_elements.append(plt.scatter([], [], s=150, c='gray', edgecolors='black', linewidths=2, label='KHC (highlighted)'))
    for ticker in sorted(other_tickers):
        legend_elements.append(plt.scatter([], [], s=60, c=color_map[ticker], edgecolors='white', label=ticker))

    plt.legend(handles=legend_elements, loc='best', fontsize=10, markerscale=1.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    output_path = 'data/pca_scatter.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")

    plt.close()

    return df_viz, pca

if __name__ == "__main__":
    df_result, pca_model = pca_visualization()
    if df_result is not None:
        print(f"\n=== KHC DATA POINTS ===")
        khc_data = df_result[df_result['ticker'] == 'KHC']
        print(f"KHC has {len(khc_data)} data points in the PCA visualization")
        if len(khc_data) > 0:
            print(f"KHC years covered: {sorted(khc_data['fiscal_year'].unique())}")