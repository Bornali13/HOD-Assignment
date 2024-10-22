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

# Create interaction terms between screen time and gender
formula = 'avg_wellbeing_log ~ C_we_log * gender + C_wk_log * gender + G_we_log * gender + G_wk_log * gender + S_we_log * gender + S_wk_log * gender + T_we_log * gender + T_wk_log * gender'

# Fit the GLM or OLS model
interaction_model = smf.ols(formula=formula, data=merged_df).fit()

# Print the summary of the regression model
print(interaction_model.summary())