#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Read your CSV
    df = pd.read_csv('BSA_ODP_PCA_REGIONAL_DRUG_SUMMARY.csv')

    # Top 10 by total ITEMS
    drug_items = df.groupby('BNF_CHEMICAL_SUBSTANCE')['ITEMS'].sum().sort_values(ascending=False)
    top_10_drugs_items = drug_items.head(10)

    plt.figure(figsize=(6,4))
    plt.barh(top_10_drugs_items.index, top_10_drugs_items.values, color='darkcyan')
    plt.title('Top 10 Antidepressants by Items (All Years)', fontsize=12)
    plt.xlabel('Total Items')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('top_10_items_barh.png')
    plt.show()

    # Top 10 by total COST
    drug_cost = df.groupby('BNF_CHEMICAL_SUBSTANCE')['COST'].sum().sort_values(ascending=False)
    top_10_drugs_cost = drug_cost.head(10)

    plt.figure(figsize=(6,4))
    plt.barh(top_10_drugs_cost.index, top_10_drugs_cost.values, color='firebrick')
    plt.title('Top 10 Antidepressants by Total Cost (All Years)', fontsize=12)
    plt.xlabel('Total Cost (£)')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('top_10_cost_barh.png')
    plt.show()

if __name__ == "__main__":
    main()
