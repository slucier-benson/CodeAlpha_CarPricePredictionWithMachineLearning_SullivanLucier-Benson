import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load and preprocess data
data = pd.read_csv('car data.csv')
data = data.drop('Car_Name', axis=1)
data['Car_Age'] = 2023 - data['Year']
data = data.drop('Year', axis=1)
data = pd.get_dummies(data, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)

# Define features and target
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: Train initial model
initial_model = RandomForestRegressor(random_state=42)
initial_model.fit(X_train, y_train)
y_pred_initial = initial_model.predict(X_test)

# Calculate initial metrics
mae_initial = mean_absolute_error(y_test, y_pred_initial)
r2_initial = r2_score(y_test, y_pred_initial)
print("Initial Model - Mean Absolute Error:", mae_initial)
print("Initial Model - R² Score:", r2_initial)

# Visualize initial predictions with units labeled
plt.scatter(y_test, y_pred_initial)
plt.xlabel('Actual Prices (Lakhs of Rupees)')
plt.ylabel('Predicted Prices (Lakhs of Rupees)')
plt.title('Initial Model: Actual vs Predicted Car Prices')
plt.show()

# Step 2: Hyperparameter tuning with Randomized Search
param_grid = {
    'n_estimators': [500],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['log2'],
    'max_depth': [10],
    'bootstrap': [False]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_grid,
    n_iter=1,  # Set to 1 since there's only one combination in param_grid
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42,
    scoring='neg_mean_absolute_error'
)

random_search.fit(X_train, y_train)

# Get the best parameters
best_params = random_search.best_params_
print("Best parameters from Random Search:", best_params)
print("Best MAE from Random Search:", -random_search.best_score_)

# Step 3: Re-train model with best parameters
optimized_model = RandomForestRegressor(**best_params, random_state=42)
optimized_model.fit(X_train, y_train)
y_pred_optimized = optimized_model.predict(X_test)

# Calculate metrics for the optimized model
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)
print("Optimized Model - Mean Absolute Error:", mae_optimized)
print("Optimized Model - R² Score:", r2_optimized)

# Visualize optimized predictions with units labeled
plt.scatter(y_test, y_pred_optimized)
plt.xlabel('Actual Prices (Lakhs of Rupees)')
plt.ylabel('Predicted Prices (Lakhs of Rupees)')
plt.title('Optimized Model: Actual vs Predicted Car Prices')
plt.show()

# Optional: Review the data structure and basic statistics
print(data.head())
print(data.info())
print(data.describe())
