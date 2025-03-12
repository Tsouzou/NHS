#!/usr/bin/env python3
import pandas as pd

def display_colored_tables(df):
    """
    Displays pivot tables for 'ITEMS' and 'COST' with a global color gradient.
    Note: This requires a rich display environment (e.g., Jupyter Notebook) 
    to render the colored HTML tables.
    """
    # Group data
    annual_items_region = df.groupby(['YEAR','REGION_NAME'], as_index=False)['ITEMS'].sum()
    items_pivot = annual_items_region.pivot(index='YEAR', columns='REGION_NAME', values='ITEMS')
    
    annual_cost_region = df.groupby(['YEAR','REGION_NAME'], as_index=False)['COST'].sum()
    cost_pivot = annual_cost_region.pivot(index='YEAR', columns='REGION_NAME', values='COST')
    
    # For display in a Jupyter notebook or similar
    from IPython.display import display
    
    # ITEMS PIVOT with global color scaling (axis=None)
    styled_items = (
        items_pivot.style
        .background_gradient(cmap='Blues', axis=None)   # global color scale
        .format('{:,.0f}')                              # integer w/ commans
        .set_caption("Annual Antidepressant Items per Region (Global Scale)")
    )
    display(styled_items)
    
    # COST PIVOT with global color scaling (axis=None)
    styled_cost = (
        cost_pivot.style
        .background_gradient(cmap='Oranges', axis=None) # global color scale
        .format('{:,.2f}')                              # 2 decimal places
        .set_caption("Annual Antidepressant Cost per Region (Global Scale)")
    )
    display(styled_cost)

def main():
    df = pd.read_csv('BSA_ODP_PCA_REGIONAL_DRUG_SUMMARY.csv')
    display_colored_tables(df)

if __name__ == "__main__":
    main()
