import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

chunk_size = 15000  # Adjust the chunk size as needed
chunks = pd.read_csv('C:/Users/Asus/OneDrive/Documents/GitHub/HOD-Assignment/merged_dataset.csv', encoding='utf-8', chunksize=chunk_size)
merged_df = pd.concat(chunks)

# Modify the formula to include demographic variables (assuming they are already binary-coded)
# Example: gender (0 for male, 1 for female), minority (0 for majority, 1 for minority), deprived (0 for not deprived, 1 for deprived)
formula = 'avg_wellbeing_log ~ C_we_log + C_wk_log + G_we_log + G_wk_log + S_we_log + S_wk_log + T_we_log + T_wk_log + gender + minority + deprived'

# Fit the GLM model with demographic variables
glm_model = smf.glm(formula=formula, data=merged_df, family=sm.families.Gaussian()).fit()

# Print the summary
print(glm_model.summary())
