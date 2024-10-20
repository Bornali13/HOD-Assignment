
import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset (merged_df)
chunk_size = 15000
chunks = pd.read_csv('C:/Users/Asus/OneDrive/Documents/GitHub/HOD-Assignment/merged_dataset.csv', encoding='utf-8', chunksize=chunk_size)
merged_df = pd.concat(chunks)

# Define your variables (e.g., avg_wellbeing_log as the dependent variable and screen time variables as predictors)
X = merged_df[['C_we_log', 'C_wk_log', 'G_we_log', 'G_wk_log', 'S_we_log', 'S_wk_log', 'T_we_log', 'T_wk_log']]
y = merged_df['avg_wellbeing_log']

# Start the Bayesian Model
with pm.Model() as model:
    # Priors for the coefficients (assuming normal distribution)
    intercept = pm.Normal('Intercept', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=1, shape=X.shape[1])
    sigma = pm.HalfNormal('sigma', sigma=1)
    
    # Expected value of the dependent variable
    mu = intercept + pm.math.dot(X, beta)
    
    # Likelihood (data likelihood based on observed data)
    likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)
    
    # Inference
    trace = pm.sample(2000, return_inferencedata=True)  # Sampling from the posterior
    
    # Posterior summary
    pm.plot_trace(trace)
    pm.summary(trace)

# Plot the results
plt.show()

# Analyze the posterior
