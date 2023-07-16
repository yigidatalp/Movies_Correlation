# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 08:48:35 2023

@author: Yigitalp
"""

# Import libraries
from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Adjust the size of plots
matplotlib.rcParams['figure.figsize'] = (12, 8)

# Read in the data
df = pd.read_csv('movies.csv')

# Look at the data (the first 5 rows/records)
df_head = df.head()

# Check whether there is any missing data and if so cleam them
df.info()
df_notnull = df.dropna()
df_notnull.info()

# Convert data types from float to integer where possible
df_notnull['votes'] = df_notnull['votes'].astype('int64')
df_notnull['budget'] = df_notnull['budget'].astype('int64')
df_notnull['gross'] = df_notnull['gross'].astype('int64')
df_notnull['runtime'] = df_notnull['runtime'].astype('int64')
df_notnull.info()


# Fetch year info from released column
first_step = df_notnull['released'].astype(str).str.split(' ', expand=True)
second_step = pd.DataFrame(first_step.loc[:, 2])
second_step.columns = ['yearcorrect']
third_step = second_step[second_step['yearcorrect'].str.isnumeric() == True]
third_step['yearcorrect'] = third_step['yearcorrect'].astype('int64')
third_step.info()

# Create yearcorrect column (dropping year column is optional)
df_notnull['yearcorrect'] = third_step['yearcorrect']
df_notnull['yearcorrect'] = df_notnull['yearcorrect'].fillna(
    df_notnull['year'])
df_notnull['yearcorrect'] = df_notnull['yearcorrect'].astype('int64')
df_notnull = df_notnull.drop('year', axis=1)
df_notnull.info()

# Order by gross
df_notnull = df_notnull.sort_values(by=['gross'], ascending=False)

# Drop any duplicates
df_notnull_unique = df_notnull.drop_duplicates()

# Correlation matrix heatmap
correlation_matrix = df_notnull_unique.corr()  # pearson, kendall, spearman
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()

# %%
# Scatter plot budget vs gross
plt.scatter(x=df_notnull_unique['budget'], y=df_notnull_unique['gross'])
plt.title('Budget vs. Gross Earnings')
plt.xlabel('Budget for Film')
plt.ylabel('Gross Earnings')
plt.show()

# %%
# Reg plot budget vs gross
sns.regplot(data=df_notnull_unique,  x='budget', y='gross',
            scatter_kws={'color': 'red'}, line_kws={'color': 'blue'})

# %%
# Convert objects into numeric represantations:
df_numerized = df_notnull_unique
for col_name in df_numerized.columns:
    if (df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes

# Normalize features
df_numerized_normalized = (
    df_numerized - df_numerized.min()) / (df_numerized.max()-df_numerized.min())

# Correlation matrix heatmap (all features included)
correlation_matrix_normalized = df_numerized_normalized.corr()
sns.heatmap(correlation_matrix_normalized, annot=True)
plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()

# Unstack correlation matrix
correlation_pairs = correlation_matrix_normalized.unstack()
correlation_pairs_df = pd.DataFrame(correlation_pairs, columns=['correl'])
correlation_pairs_df = correlation_pairs_df[correlation_pairs_df['correl'] != 1]
correlation_pairs_df = correlation_pairs_df.sort_values(
    by='correl', ascending=False)
correlation_pairs_df = correlation_pairs_df.drop_duplicates()
positive_correl = correlation_pairs_df[correlation_pairs_df['correl'] >= 0.5]
negative_correl = correlation_pairs_df[correlation_pairs_df['correl'] <= -0.5]
