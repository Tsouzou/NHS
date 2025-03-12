#!/usr/bin/env python3
import pandas as pd

def main():
    df = pd.read_csv('BSA_ODP_PCA_REGIONAL_DRUG_SUMMARY.csv')
    
    # TABLE: ITEMS PER REGION
    annual_items_region = df.groupby(['YEAR','REGION_NAME'], as_index=False)['ITEMS'].sum()
    items_pivot = annual_items_region.pivot(index='YEAR', columns='REGION_NAME', values='ITEMS')
    print("\nTotal Annual Antidepressant Items per Region\n")
    print(items_pivot)

    # TABLE: COST PER REGION
    annual_cost_region = df.groupby(['YEAR','REGION_NAME'], as_index=False)['COST'].sum()
    cost_pivot = annual_cost_region.pivot(index='YEAR', columns='REGION_NAME', values='COST')
    print("\nTotal Annual Antidepressant Cost per Region\n")
    print(cost_pivot)

    print("\nTables generated successfully.\n")

if __name__ == "__main__":
    main()
