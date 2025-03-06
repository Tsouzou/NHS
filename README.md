Below is a concise list of the exercises mentioned in your notebook, organized by the three learning sections. These correspond to the tasks you'll need to complete for data manipulation, visualization, and metrics/insights.

---

## Part 1: **Transforming and Aggregating Data**

1. **Question 1**  
   Nationally, calculate the top 10 prescribed antidepressants (by number of items) across the *entire* time frame, sorted from largest to smallest.

2. **Question 2**  
   Calculate the **monthly national cost** of Mirtazapine prescribing.  
   *(Hint: You’ll need to group or filter for Mirtazapine, then aggregate by “month” or “YEAR_MONTH.”)*

3. **Question 3**  
   What is the **annual spend** on Sertraline hydrochloride prescribing **in the Midlands region**?  
   *(Hint: You’ll need to group by “region,” filter for Midlands and Sertraline, then aggregate by “year.”)*

---

## Part 2: **Data Visualization**

1. **Question 1**  
   Create a **horizontal bar chart** of the **top 5 most prescribed** drugs in **2024**, arranged in descending order (largest to smallest).

2. **Question 2**  
   Create a **vertical bar chart** showing the **total annual cost** of Sertraline hydrochloride prescribing in the **NORTH WEST** region.

3. **Question 3**  
   Create a **line chart** of the *national* **monthly cost** (rounded to the nearest pound) of **escitalopram**.

---

## Part 3: **Data Metrics and Insights**

> _These use the dataframe `pca_regional_drug_summary_df` mentioned in the instructions._

1. **Question 1**  
   Create a **monthly line chart** showing the **total national prescribing cost** (all drugs combined).

2. **Question 2**  
   Create **annual summary statistics** (min, Q1, median, Q3, max) for the **national monthly prescribing cost**.  
   *(Hint: Summarize monthly cost first, then compute these descriptive statistics per year.)*

3. **Question 3**  
   Create a **grouped boxplot** for the statistics in Question 2. This will show four boxplots total (one per year).

4. **Question 4**  
   Calculate the **annual mean monthly total national prescribing cost** and display it in a **vertical bar chart**.

5. **Question 5**  
   Create a **monthly line chart** showing the **percentage of national prescribing** that is from the **'02: Cardiovascular System'** BNF_CHAPTER.  
   *(Note: This example shows how to calculate and plot a proportion of the total; the instructions reference cardiovascular prescribing, but the same concept applies to any coordinate data you choose.)*

6. **Question 6**  
   Create a **pivoted table** that shows the cost of anti-depressant prescribing **per region per year**.  
   - Each row should represent a year.  
   - Each column should be a region.  
   - The values are the *summed cost* of prescribing.

---

### How to Proceed

- Use basic pandas operations (filtering, grouping, aggregating) for **Part 1**.  
- Explore matplotlib, pandas’ built-in `.plot()`, or seaborn for **Part 2** visualizations.  
- For **Part 3**, consider descriptive statistics methods in pandas (e.g., `.describe()`, `.mean()`, `.min()`, `.max()`, etc.) and pivot tables.  

Completing these exercises will build your skills in data wrangling, visualization, and higher-level insights—ultimately preparing you to write your final report.
