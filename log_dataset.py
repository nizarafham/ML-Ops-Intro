import pandas as pd
import xgboost
import mlflow
from mlflow.data.pandas_dataset import PandasDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
dataset_source_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv"
raw_data = pd.read_csv(dataset_source_url, delimiter=";")

# Extract features and target
y = raw_data["quality"]
X = raw_data.drop("quality", axis=1)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=17
)

# Encode the target variable
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Train the model
model = xgboost.XGBClassifier()
model.fit(X_train, y_train_encoded)

# Make predictions
y_test_pred = model.predict(X_test)

# Create evaluation dataset
eval_data = X_test.copy()
eval_data["label"] = y_test
eval_data["predictions"] = le.inverse_transform(y_test_pred)

# Create PandasDataset for evaluation
pd_dataset = mlflow.data.from_pandas(
    eval_data, 
    predictions="predictions", 
    targets="label"
)

# Set experiment name
mlflow.set_experiment("White Wine Quality")

# Start MLflow run
with mlflow.start_run() as run:
    # Log the input dataset
    mlflow.log_input(pd_dataset, context="training")
    
    # Log model parameters
    mlflow.log_params({
        'max_depth': model.max_depth,
        'learning_rate': model.learning_rate,
        'n_estimators': model.n_estimators
    })
    
    # Log the model
    mlflow.xgboost.log_model(
        artifact_path="white-wine-xgb",
        xgb_model=model,
        input_example=X_test
    )
    
    # Run evaluation
    result = mlflow.evaluate(
        data=pd_dataset,
        predictions=None,
        model_type="classifier"
    )
    
    print("Evaluation Results:")
    print(result.metrics)