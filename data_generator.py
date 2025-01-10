import pandas as pd
import numpy as np

def generate_data(n_samples, drift_params=None):
    """Generate synthetic customer churn data"""
    # np.random.seed(42)
    
    data = {
        'usage_mins': np.random.normal(600, 100, n_samples),
        'monthly_bill': np.random.normal(70, 20, n_samples),
        'support_calls': np.random.poisson(3, n_samples)
    }
    
    # Apply drift if specified
    if drift_params:
        for feature, (shift, scale) in drift_params.items():
            data[feature] = data[feature] * scale + shift
    
    df = pd.DataFrame(data)
    
    # Generate target (churn probability affected by features)
    churn_prob = 1 / (1 + np.exp(-(
        -5 + 
        0.005 * df['usage_mins'] + 
        0.02 * df['monthly_bill'] +
        0.2 * df['support_calls']
    )))
    df['churn'] = (np.random.random(n_samples) < churn_prob).astype(int)
    
    return df

# Generate datasets
if __name__ == "__main__":
    # Initial training data
    train_data = generate_data(1000)
    train_data.to_csv('data/train.csv', index=False)
    
    # New data with drift
    drift_params = {
        'usage_mins': (5, 1.5),    # Increase mean and variance
        'monthly_bill': (2, 1.6),   # Slight increase
        'support_calls': (0.5, 1.1)    # More variance in support calls
    }
    new_data = generate_data(1000, drift_params)
    new_data.to_csv('data/new_data.csv', index=False)