import pandas as pd
import numpy as np
import mlflow

def create_bins(data, n_bins=10):
    """Create bins using percentile method"""
    print(n_bins)
    bins = np.percentile(data, np.linspace(0, 100, n_bins + 1))
    return bins

def get_distributions(expected, actual, bins):
    """Calculate % distribution for each dataset"""
    expected_dist = np.histogram(expected, bins)[0] / len(expected)
    actual_dist = np.histogram(actual, bins)[0] / len(actual)
    
    # Add small epsilon to avoid division by zero
    expected_dist = np.clip(expected_dist, 1e-8, None)
    actual_dist = np.clip(actual_dist, 1e-8, None)
    
    return expected_dist, actual_dist

def calculate_psi(expected_dist, actual_dist):
    """Calculate PSI value"""
    print(actual_dist)
    print(expected_dist)
    print(actual_dist - expected_dist)
    print(np.log(actual_dist / expected_dist))

    psi = np.sum(
        (actual_dist - expected_dist) * 
        np.log(actual_dist / expected_dist)
    )
    return psi

# def detect_drift_psi(expected_dataset, actual_dataset, n_bins=10):
#     """Complete PSI calculation  for each feature"""
#     drift_metrics = {}
#     for column in expected_dataset.columns:
#         if column != 'churn':
#             expected_data = expected_dataset[column]
#             actual_data = actual_dataset[column]

#             bins = create_bins(expected_data, n_bins)
            
#             # Get distributions
#             expected_dist, actual_dist = get_distributions(
#                 expected_data, actual_data, bins
#             )
            
#             # Calculate PSI
#             psi_value = calculate_psi(expected_dist, actual_dist)

#             drift_metrics[f"{column}_psi"] = psi_value
            
#     # Determine if drift is significant (PSI > 0.2 is considered significant)
#     drift_check = any(v > 0.1 and v <= 0.2 for v in drift_metrics.values())
#     mlflow.log_param("drift_needs_check", drift_check)
    
#     drift_detected = any(v > 0.2 for v in drift_metrics.values())
#     mlflow.log_param("drift_detected", drift_detected)
    
#     return drift_metrics, drift_detected


def calculate_psi_column(expected_data, actual_data, column_name, n_bins=10):
    """Complete PSI calculation  for each feature"""

    bins = create_bins(expected_data, n_bins)
    
    # Get distributions
    expected_dist, actual_dist = get_distributions(
        expected_data, actual_data, bins
    )
    
    # Calculate PSI
    psi_value = calculate_psi(expected_dist, actual_dist)
    
    return psi_value, bins, expected_dist, actual_dist

def train_and_monitor_drift():
    # Start MLflow run
    with mlflow.start_run(run_name="model_with_drift_detection") as run:
              # Load data
        train_data = pd.read_csv('data/train.csv')
        new_data = pd.read_csv('data/new_data.csv')
        
        # Calculate and log PSI for each feature
        drift_metrics = {}
        drift_detected = False
        
        for column in train_data.columns:
            psi_value, bins, expected_dist, actual_dist = calculate_psi_column(
                train_data[column], 
                new_data[column],
                column
            )
            
            # Log PSI value
            drift_metrics[f"psi_{column}"] = psi_value
            
            # Check if drift detected
            if psi_value >= 0.1:
                drift_detected = True
                
            # Log distribution data as artifacts
            dist_df = pd.DataFrame({
                'bins': bins[:-1],
                'expected_dist': expected_dist,
                'actual_dist': actual_dist
            })
            dist_df.to_csv(f'distribution_{column}.csv', index=False)
            mlflow.log_artifact(f'distribution_{column}.csv')
        
        # Log overall drift metrics
        mlflow.log_metrics(drift_metrics)
        
        # Log drift status as parameter
        mlflow.log_param("drift_detected", drift_detected)
        
        return run.info.run_id, drift_metrics, drift_detected


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("drift_detection_demo")
    
    run_id, drift_metrics, has_drift = train_and_monitor_drift()
    
    print("\nDrift Detection Results:")
    print("-" * 50)
    print(f"Run ID: {run_id}")
    print("\nPSI Values:")
    for metric, value in drift_metrics.items():
        print(f"{metric}: {value:.4f}")
    print(f"\nDrift Detected: {has_drift}")