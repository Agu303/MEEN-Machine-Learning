import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import joblib
import json
import time

# 1. Load and process training data w/ error handling
try:
    print("Loading training data...")
    with open('train_data.json', 'r') as f:
        data = json.load(f)
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: Could not find train_data.json")
    exit()
except Exception as e:
    print("Error loading data:", str(e))
    exit()

def extract_features_labels(data):
    features = []
    labels = []
    for sample in data:
        interface_prop = sample['interface_prop']
        strain = sample['strain']
        stress = sample['stress']
        for i in range(len(strain)):
            features.append(np.append(interface_prop, strain[i]))
            labels.append(stress[i])
    return np.vstack(features), np.array(labels)

def split_data(data, val_split=0.2):
    np.random.seed(42)
    np.random.shuffle(data)
    val_size = int(val_split*len(data))
    val_data = data[:val_size]
    train_data = data[val_size:]
    return train_data, val_data

train_data, val_data = split_data(data)
X_train, y_train = extract_features_labels(train_data)
X_val, y_val = extract_features_labels(val_data)

##########
DEBUG_MODE = False
DEBUG_SAMPLES = 10000  

if DEBUG_MODE:
    X_train = X_train[:DEBUG_SAMPLES]
    y_train = y_train[:DEBUG_SAMPLES]
    print(f" 😮 DEBUG MODE ON: Using only {DEBUG_SAMPLES} samples for training. 😮")
#########

models = {
    'rf': RandomForestRegressor(n_estimators=100, random_state=42),
    'xgb': XGBRegressor(n_estimators=100, random_state=42)
}

if not DEBUG_MODE:
    print("\n⏱️ Starting 5-fold cross-validation 😎:")
    for name, model in models.items():
        start = time.time()
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
        end = time.time()
        print(f"{name} - R² scores for each fold: {scores}")
        print(f"{name} - Mean R² Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        print(f"{name} - Time taken: {(end - start)/60:.2f} minutes")


trained_models = {}
model_predictions = {}
for name, model in models.items():
    print(f"\nTraining final {name} model...")
    trained_models[name] = model.fit(X_train, y_train)
    
    # Make predictions on validation set
    predictions = model.predict(X_val)
    model_predictions[name] = predictions
    
    # Calculate metrics
    r2 = r2_score(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    
    print(f"{name} - Validation R² Score: {r2:.4f}")
    print(f"{name} - Validation RMSE: {rmse:.4f}")
    
    # Save model
    joblib.dump(model, f'{name}_model.pkl')

def combined_predict(X):
    predictions = []
    for model in trained_models.values():
        predictions.append(model.predict(X))
    return np.mean(predictions, axis=0)

combined_predictions = np.mean([preds for preds in model_predictions.values()], axis=0)
combined_r2 = r2_score(y_val, combined_predictions)
combined_rmse = np.sqrt(mean_squared_error(y_val, combined_predictions))

print("\nCombined Model Performance on Validation Set:")
print(f"Combined Model R² Score: {combined_r2:.4f}")
print(f"Combined Model RMSE: {combined_rmse:.4f}")

print("\nTraining complete! All models saved.")
