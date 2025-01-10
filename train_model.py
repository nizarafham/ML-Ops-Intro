import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle

def train_and_log_model(data_path, str_model_type):
    # Load and prepare data
    data = pd.read_csv(data_path)
    X = data.drop('churn', axis=1)
    y = data['churn']
    # Start MLflow run
    with mlflow.start_run(run_name=f"synthetic_churn_model_{str_model_type}") as run:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        if str_model_type == 'RandomForest':
             
            # Define and train model
            params = {
                'n_estimators': 100,
                'max_depth': 5,
                'random_state': 42
            }
            
            model = RandomForestClassifier(**params)
        else:
            params = {}
            model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc_score': roc_auc_score(y_test, y_prob)
        }
        
        # Log parameters, metrics, and model
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            'model_type':str_model_type
        })
        mlflow.sklearn.log_model(model, f"model {str_model_type}")
        
        # Save feature names for drift detection
        feature_names = X.columns.tolist()
        mlflow.log_dict({'feature_names': feature_names}, 'feature_names.json')
        
        return run.info.run_id


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("churn_prediction")
    run_id_rf = train_and_log_model("data/train.csv",'RandomForest')
    run_id_lr = train_and_log_model("data/train.csv",'LogisticRegression')
    print(f"Model trained and logged with run_id: {run_id_rf}")
    print(f"Model trained and logged with run_id: {run_id_lr}")