import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.formula.api import ols


chunk_size = 15000  # Adjust the chunk size as needed
chunks = pd.read_csv('C:/Users/Asus/OneDrive/Documents/GitHub/HOD-Assignment/merged_dataset.csv', encoding='utf-8', chunksize=chunk_size)
merged_df = pd.concat(chunks)

###Univariate Analysis###

#Screentime
screentime = merged_df[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']]

# Calculate the median for each screen time variable
median_values = screentime.median()

# Plot the bar chart for the median values
plt.figure(figsize=(10, 6))
median_values.plot(kind='bar', color='lightblue')

# Add labels and title
plt.title('Median Values of Screen Time Variables')
plt.xlabel('Screen Time Variables')
plt.ylabel('Median Value')

# Display the median values on top of each bar
for index, value in enumerate(median_values):
    plt.text(index, value + 0.1, f'{value:.2f}', ha='center')

# Save the plot as an image
plt.savefig('screentime_median_values.png')

# Show the plot
plt.show()


screentime_log = merged_df[['C_we_log', 'C_wk_log', 'G_we_log', 'G_wk_log', 'S_we_log', 'S_wk_log', 'T_we_log', 'T_wk_log']]
import warnings
warnings.filterwarnings('ignore')
# Create a boxplot of all screentime variables in one figure
plt.figure(figsize=(12, 6))
sns.boxplot(data=screentime_log)
plt.title('Boxplot of Screen Time Variables')
plt.ylabel('Log Transformed Screen Time')
plt.xticks(rotation=45)
#save
plt.savefig('Boxplot of Screen Time Variables.png')
# Show the plot
plt.tight_layout()
plt.show()


#wellbeing
import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(merged_df['avg_wellbeing_log'], bins=7)
plt.subplot(1,2,2)
sns.boxplot(merged_df['avg_wellbeing_log'])
plt.savefig('Hist_Boxplot of wellbeing.png')
plt.show()


###Bivariate Analysis


#Wellbeing and screentime
from scipy.stats import spearmanr

# Calculate pearson correlation 
# Select the columns to compute correlations (e.g., log-transformed screentime and well-being)
columns_for_correlation = ['C_we_log', 'C_wk_log', 'G_we_log', 'G_wk_log', 'S_we_log', 'S_wk_log', 'T_we_log', 'T_wk_log', 'avg_wellbeing_log']

# Calculate the Pearson correlation matrix
correlation_matrix = merged_df[columns_for_correlation].corr(method='pearson')

# Display the correlation matrix
print(correlation_matrix)

# Optionally, visualize the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Pearson Correlation Matrix')
# Save the plot as an image
plt.savefig(f'Pearson correlation matrix.png')
plt.show()


#Wellbeing and Demographic data

model = ols('avg_wellbeing ~ C(gender) + C(minority) + C(deprived)', data=merged_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

#Screentime and Demographic Data
def plot_boxplots_for_demographic(data, demographic_col):
    screentime_log = merged_df[['C_we_log', 'C_wk_log', 'G_we_log', 'G_wk_log', 'S_we_log', 'S_wk_log', 'T_we_log', 'T_wk_log']]
    
    # Plot boxplots
    plt.figure(figsize=(14, 8))
    for i, col in enumerate(screentime_log):
        plt.subplot(2, 4, i+1)
        sns.boxplot(x=data[demographic_col], y=data[col])
        plt.title(f'{col} by {demographic_col}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'Boxplot of {demographic_col} and Screentime')
    plt.show()

# For gender
plot_boxplots_for_demographic(merged_df, 'gender')

# For minority status
plot_boxplots_for_demographic(merged_df, 'minority')

# For deprived status
plot_boxplots_for_demographic(merged_df, 'deprived')