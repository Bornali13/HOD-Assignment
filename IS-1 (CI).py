import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats

# Load your dataset (merged_df)
chunk_size = 15000
chunks = pd.read_csv('C:/Users/Asus/OneDrive/Documents/GitHub/HOD-Assignment/merged_dataset.csv', encoding='utf-8', chunksize=chunk_size)
merged_df = pd.concat(chunks)

##Screentime
screentime_log = merged_df[['C_we_log', 'C_wk_log', 'G_we_log', 'G_wk_log', 'S_we_log', 'S_wk_log', 'T_we_log', 'T_wk_log']]

# Function to compute bootstrap confidence interval for the median
def bootstrap_median_ci(data, n_bootstrap=1000, ci_percentile=95):
    medians = []
    # Perform bootstrap resampling
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        medians.append(np.median(sample))
    
    # Compute percentiles for confidence interval
    lower_percentile = (100 - ci_percentile) / 2
    upper_percentile = 100 - lower_percentile
    return np.percentile(medians, [lower_percentile, upper_percentile])

# Apply this to each column of screentime data
ci_results = {}

for column in screentime_log.columns:
    # Compute confidence interval for log-transformed data
    median_ci_log = bootstrap_median_ci(screentime_log[column].dropna(), n_bootstrap=1000, ci_percentile=95)
    
    # Convert the log confidence intervals back to the original scale
    median_ci_original = np.exp(median_ci_log)  # Apply exp to get back to original scale
    
    ci_results[column] = median_ci_original

# Display the results
for column, ci in ci_results.items():
    print(f"95% Confidence Interval for the median of {column} (original scale): {ci}")
    

##Wellbeing

# Remove NaN values from the log_wellbeing data
log_wellbeing_cleaned = merged_df['avg_wellbeing_log'].dropna()

# Calculate the mean and standard error of the mean (SEM)
mean_log_wellbeing = np.mean(log_wellbeing_cleaned)
sem_log_wellbeing = stats.sem(log_wellbeing_cleaned)

# 95% confidence interval for the log-transformed data
confidence_interval_log = stats.norm.interval(0.95, loc=mean_log_wellbeing, scale=sem_log_wellbeing)

# Display the confidence interval in the log scale
print(f"95% Confidence Interval (log scale): {confidence_interval_log}")
# Convert the log scale confidence interval back to the original scale
confidence_interval_original = np.exp(confidence_interval_log)

# Display the confidence interval in the original scale
print(f"95% Confidence Interval (original scale): {confidence_interval_original}")