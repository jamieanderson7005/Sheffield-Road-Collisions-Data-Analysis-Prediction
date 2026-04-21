import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

def run_regression_analysis():
    print("Starting regression analysis")
    df = pd.read_csv('cleaned_data.csv')

    df['target_log'] = np.log1p(df['number_of_vehicles'])
    features = [
        'weather_conditions', 'light_conditions', 'road_surface_conditions', 'urban_or_rural_area', 'hour', 'speed_limit', 'day_of_week'
    ]

    x = df[features]
    y = df['target_log']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    model.fit(x_train, y_train)
    
    pred_log = model.predict(x_test)
    y_test_original = np.expm1(y_test)
    preds_original = np.expm1(pred_log)

    mse = mean_squared_error(y_test_original, preds_original)
    r2 = r2_score(y_test_original, preds_original)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score (Variance): {r2:.4f}")

    plt.figure(figsize=(8,6))
    plt.scatter(y_test_original, preds_original, alpha=0.3, color='green')
    plt.plot([y_test_original.min(), y_test_original.max()],
             [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
    plt.title('Actual vs predicted number of vehicles')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig('regression_accuracy_visual.png')

    print("Plot saved")

if __name__ == "__main__":
    run_regression_analysis()