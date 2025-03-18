import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import joblib

# 1. Load training data w/ error handling
try:
    print("Loading training data...")
    xs_train = np.load('xs_train.npy')
    ys_train = np.load('ys_train.npy')
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: Could not find training data files (xs_train.npy or ys_train.npy)")
    exit()
except Exception as e:
    print("Error loading data:", str(e))
    exit()

# 2. Define models
models = {
    'rf': RandomForestRegressor(n_estimators=100, random_state=42),
    'xgb': XGBRegressor(n_estimators=100, random_state=42),
    'lgbm': LGBMRegressor(n_estimators=100, random_state=42)
}

# 3. Compare models using cross-validation
print("\nComparing models using 5-fold cross-validation:")
for name, model in models.items():
    scores = cross_val_score(model, xs_train, ys_train, cv=5, scoring='r2')
    print("%s - R² scores for each fold: %s" % (name, scores))
    print("%s - Mean R² Score: %.4f (+/- %.4f)" % (name, scores.mean(), scores.std() * 2))

# 4. Train and save the models
trained_models = {}
model_predictions = {}
for name, model in models.items():
    print("\nTraining final %s model..." % name)
    trained_models[name] = model.fit(xs_train, ys_train)
    
    # Make predictions
    predictions = model.predict(xs_train)
    model_predictions[name] = predictions
    
    # Calculate metrics
    r2 = r2_score(ys_train, predictions)
    rmse = np.sqrt(mean_squared_error(ys_train, predictions))
    
    print("%s - Final R² Score: %.4f" % (name, r2))
    print("%s - Final RMSE: %.4f" % (name, rmse))
    
    # Save model
    joblib.dump(model, name + '_model.pkl')

# 5. Create and save combined model
def combined_predict(X):
    predictions = []
    for model in trained_models.values():
        predictions.append(model.predict(X))
    return np.mean(predictions, axis=0)

# Calculate combined model performance
combined_predictions = np.mean([preds for preds in model_predictions.values()], axis=0)
combined_r2 = r2_score(ys_train, combined_predictions)
combined_rmse = np.sqrt(mean_squared_error(ys_train, combined_predictions))

print("\nCombined Model Performance:")
print("Combined Model R² Score: %.4f" % combined_r2)
print("Combined Model RMSE: %.4f" % combined_rmse)

print("\nTraining complete! All models saved.")