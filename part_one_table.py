#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_colored_table(df, cmap, value_fmt, title):
    """
    Creates a heatmap-like plot for a pivoted DataFrame, with numeric labels.
    Automatically saves the figure to a PDF using the plot title as the filename.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Convert DataFrame to a 2D array suitable for imshow
    data_array = df.values
    
    # Create a heatmap
    im = ax.imshow(data_array, cmap=cmap, aspect='auto')
    
    # Optional colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Value")

    # Show row/column labels
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticklabels(df.index)

    # Annotate each cell
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = data_array[i, j]
            if pd.notnull(val):
                ax.text(
                    j, i, value_fmt.format(val),
                    ha="center", va="center", color="black"
                )
    
    ax.set_title(title)
    ax.set_xlabel(df.columns.name if df.columns.name else "")
    ax.set_ylabel(df.index.name if df.index.name else "")
    plt.tight_layout()
    
    # Save to PDF (use the title to form the filename)
    pdf_filename = f"{title.replace(' ', '_')}.pdf"
    plt.savefig(pdf_filename)
    
    # Display the plot
    plt.show()


def display_colored_tables_as_plots(df):
    """
    Creates pivot tables for ITEMS and COST, then plots each one as 
    a heatmap-like figure with color coding and numeric labels. 
    Saves each plot as a PDF, too.
    """
    # 1) Create pivot tables
    annual_items_region = df.groupby(['YEAR','REGION_NAME'], as_index=False)['ITEMS'].sum()
    items_pivot = annual_items_region.pivot(index='YEAR', columns='REGION_NAME', values='ITEMS')
    
    annual_cost_region = df.groupby(['YEAR','REGION_NAME'], as_index=False)['COST'].sum()
    cost_pivot = annual_cost_region.pivot(index='YEAR', columns='REGION_NAME', values='COST')

    # 2) Plot the ITEMS pivot
    plot_colored_table(
        df=items_pivot,
        cmap='Blues',
        value_fmt='{:,.0f}',  # integer w/ commas
        title="Annual Antidepressant Items per Region"
    )

    # 3) Plot the COST pivot
    plot_colored_table(
        df=cost_pivot,
        cmap='Oranges',
        value_fmt='{:,.2f}',  # two decimal places
        title="Annual Antidepressant Cost per Region"
    )


def main():
    df = pd.read_csv('BSA_ODP_PCA_REGIONAL_DRUG_SUMMARY.csv')
    display_colored_tables_as_plots(df)

if __name__ == "__main__":
    main()
