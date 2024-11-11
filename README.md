# Car Price Prediction with Machine Learning

## Project Overview
This project aims to develop a predictive model to estimate car prices based on various features such as age, mileage, fuel type, and transmission. The goal is to create an accurate model for predicting car prices in lakhs of rupees, improving the prediction accuracy through hyperparameter tuning.

## Objectives
- Build a machine learning model to predict car prices based on provided features.
- Visualize the initial and optimized model results for better interpretability.
- Use hyperparameter tuning to improve the model's predictive performance.

## Data Source
The dataset used for this analysis is **car data.csv**, containing the following columns:
- **Car_Name**: The name of the car model (not used in the model).
- **Year**: The year the car was manufactured.
- **Selling_Price**: The price at which the car is being sold (target variable, in lakhs of rupees).
- **Present_Price**: The current ex-showroom price of the car.
- **Driven_kms**: The number of kilometers the car has been driven.
- **Fuel_Type**: The type of fuel used by the car (Petrol/Diesel/CNG).
- **Selling_type**: Type of seller (Dealer or Individual).
- **Transmission**: Transmission type (Manual or Automatic).
- **Owner**: The number of previous owners.

## Methodology

### Data Preprocessing
- **Feature Engineering**: Calculated car age from the Year column.
- **Encoding Categorical Variables**: Converted categorical columns like Fuel_Type, Selling_type, and Transmission into dummy variables using one-hot encoding.
- **Feature Selection**: Dropped irrelevant columns, including Car_Name and Year after creating the age feature.

### Model Selection
- Used a **Random Forest Regressor** as the primary model due to its effectiveness with tabular data and ability to handle non-linear relationships.

### Model Training and Evaluation

**Initial Model:**
- Trained a baseline model with default parameters.
- **Mean Absolute Error (MAE)**: 0.6369
- **R² Score**: 0.9595
- Visualized the initial predictions against the actual car prices.

**Hyperparameter Tuning:**
- Performed hyperparameter tuning using **RandomizedSearchCV** to find optimal values for parameters such as:
  - `n_estimators`: Number of trees in the forest.
  - `max_depth`: Maximum depth of each tree.
  - `min_samples_split` and `min_samples_leaf`: Control tree splitting and leaf node creation.
  - `max_features`: Number of features to consider when splitting a node.
  - `bootstrap`: Whether or not to bootstrap samples when building trees.

- **Best parameters found:**
    ```python
    {
        'n_estimators': 500,
        'max_depth': 10,
        'max_features': 'log2',
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'bootstrap': False
  }
    ```

**Optimized Model:**
- Trained the model with the best parameters found through hyperparameter tuning.
- **Mean Absolute Error (MAE)**: 0.5344
- **R² Score**: 0.9665
- Visualized the optimized predictions against the actual car prices.

## Findings
- **Performance Improvement**: Hyperparameter tuning resulted in a significant improvement in MAE (from 0.6369 to 0.5344) and R² Score (from 0.9595 to 0.9665).
- **Visual Insights**: Scatter plots of actual vs. predicted prices show that the optimized model aligns more closely with the actual values, indicating improved prediction accuracy.
- **Key Features**: The features `Present_Price`, `Driven_kms`, and `Car_Age` have the highest predictive power in estimating car prices.

## Instructions to Run the Project
1. **Prerequisites**: Ensure `pandas`, `matplotlib`, and `scikit-learn` are installed.
2. **Run the Python Script**: Execute `CarPrice_Prediction.py` to load the data, train the model, perform hyperparameter tuning, and visualize the results.

## Conclusion
This project successfully demonstrates the importance of hyperparameter tuning in enhancing model performance. By carefully selecting the best parameters for the Random Forest Regressor, we achieved a more accurate and robust model for car price prediction.

## Future Improvements
- Explore additional algorithms such as **Gradient Boosting** or **XGBoost** for further performance enhancement.
- Consider expanding the feature set with external data sources or additional car-specific features.
