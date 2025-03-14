#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1) Read CSV
    df = pd.read_csv('BSA_ODP_PCA_REGIONAL_DRUG_SUMMARY.csv')
    
    # 2) Monthly National Totals
    # Group data by YEAR_MONTH to get total items & cost across all regions/drugs
    monthly_totals = df.groupby('YEAR_MONTH', as_index=False)[['ITEMS','COST']].sum()
    monthly_totals = monthly_totals.sort_values('YEAR_MONTH')  # ensure chronological order
    
    # Convert YEAR_MONTH to string for x-axis labeling
    monthly_totals['YEAR_MONTH_STR'] = monthly_totals['YEAR_MONTH'].astype(str)
    
    # (A) Line Chart for Monthly Items
    plt.figure(figsize=(8,5))
    plt.plot(monthly_totals['YEAR_MONTH_STR'], monthly_totals['ITEMS'], marker='o', color='blue')
    plt.title('Monthly Total Antidepressant Items (National)', fontsize=14)
    plt.xlabel('YEAR_MONTH (YYYYMM)')
    plt.ylabel('Prescribed Items')
    plt.xticks(rotation=70)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('part_two_monthly_items.png')
    plt.show()
    
    # (B) Line Chart for Monthly Cost
    plt.figure(figsize=(8,5))
    plt.plot(monthly_totals['YEAR_MONTH_STR'], monthly_totals['COST'], marker='o', color='red')
    plt.title('Monthly Total Antidepressant Cost (National)', fontsize=14)
    plt.xlabel('YEAR_MONTH (YYYYMM)')
    plt.ylabel('Cost (£)')
    plt.xticks(rotation=70)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('part_two_monthly_cost.png')
    plt.show()

if __name__ == "__main__":
    main()
