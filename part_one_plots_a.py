#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Read your CSV
    df = pd.read_csv('BSA_ODP_PCA_REGIONAL_DRUG_SUMMARY.csv')

    # Calculate Annual Items
    annual_items = df.groupby('YEAR', as_index=False)['ITEMS'].sum()

    # Calculate Annual Cost
    annual_cost = df.groupby('YEAR', as_index=False)['COST'].sum()

    # Create a single figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))

    # Left subplot: Annual Items
    axes[0].bar(annual_items['YEAR'].astype(str), annual_items['ITEMS'], color='skyblue')
    axes[0].set_title('Total Annual Antidepressant Prescribing (Items)', fontsize=12)
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Prescribed Items')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Right subplot: Annual Cost
    axes[1].bar(annual_cost['YEAR'].astype(str), annual_cost['COST'], color='salmon')
    axes[1].set_title('Total Annual Antidepressant Prescribing (Cost)', fontsize=12)
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Total Cost (£)')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    # Save to file, then show
    plt.savefig('annual_items_cost_subplots.png')
    plt.show()


if __name__ == "__main__":
    main()
