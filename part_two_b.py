#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # ----------------------------------------------------------------
    # 1) Read CSV
    # ----------------------------------------------------------------
    df = pd.read_csv('BSA_ODP_PCA_REGIONAL_DRUG_SUMMARY.csv')
    
    # ----------------------------------------------------------------
    # 2) Identify Top-5 Drugs by Items and by Cost
    # ----------------------------------------------------------------
    drugs_by_items = df.groupby('BNF_CHEMICAL_SUBSTANCE', as_index=False)['ITEMS'].sum()
    top_5_items = (drugs_by_items.sort_values('ITEMS', ascending=False)
                              .head(5)['BNF_CHEMICAL_SUBSTANCE']
                              .tolist())
    
    drugs_by_cost = df.groupby('BNF_CHEMICAL_SUBSTANCE', as_index=False)['COST'].sum()
    top_5_cost = (drugs_by_cost.sort_values('COST', ascending=False)
                             .head(5)['BNF_CHEMICAL_SUBSTANCE']
                             .tolist())
    
    print("\nTop 5 Drugs by Total Items:", top_5_items)
    print("Top 5 Drugs by Total Cost:", top_5_cost)
    
    # ----------------------------------------------------------------
    # 3) Plot Monthly Trends for Top Drugs (Items)
    # ----------------------------------------------------------------
    top_items_df = df[df['BNF_CHEMICAL_SUBSTANCE'].isin(top_5_items)]
    monthly_items_drugs = (top_items_df.groupby(['YEAR_MONTH','BNF_CHEMICAL_SUBSTANCE'], as_index=False)['ITEMS']
                                      .sum()
                                      .sort_values('YEAR_MONTH'))
    monthly_items_drugs['YEAR_MONTH_STR'] = monthly_items_drugs['YEAR_MONTH'].astype(str)
    
    plt.figure(figsize=(9,6))
    for drug in top_5_items:
        subset = monthly_items_drugs[monthly_items_drugs['BNF_CHEMICAL_SUBSTANCE'] == drug]
        plt.plot(subset['YEAR_MONTH_STR'], subset['ITEMS'], marker='o', label=drug)
    
    plt.title('Monthly Items Trend for Top 5 Drugs (by Items)', fontsize=14)
    plt.xlabel('YEAR_MONTH (YYYYMM)')
    plt.ylabel('Items Prescribed')
    plt.xticks(rotation=70)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('part_two_top5_items_trend.png')
    plt.show()
    
    # ----------------------------------------------------------------
    # 4) Plot Monthly Trends for Top Drugs (Cost)
    # ----------------------------------------------------------------
    top_cost_df = df[df['BNF_CHEMICAL_SUBSTANCE'].isin(top_5_cost)]
    monthly_cost_drugs = (top_cost_df.groupby(['YEAR_MONTH','BNF_CHEMICAL_SUBSTANCE'], as_index=False)['COST']
                                     .sum()
                                     .sort_values('YEAR_MONTH'))
    monthly_cost_drugs['YEAR_MONTH_STR'] = monthly_cost_drugs['YEAR_MONTH'].astype(str)
    
    plt.figure(figsize=(9,6))
    for drug in top_5_cost:
        subset = monthly_cost_drugs[monthly_cost_drugs['BNF_CHEMICAL_SUBSTANCE'] == drug]
        plt.plot(subset['YEAR_MONTH_STR'], subset['COST'], marker='o', label=drug)
    
    plt.title('Monthly Cost Trend for Top 5 Drugs (by Cost)', fontsize=14)
    plt.xlabel('YEAR_MONTH (YYYYMM)')
    plt.ylabel('Cost (£)')
    plt.xticks(rotation=70)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('part_two_top5_cost_trend.png')
    plt.show()

if __name__ == "__main__":
    main()
