import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_regression_analysis():
    print("Starting regression analysis")
    df = pd.read_csv('cleaned_data.csv')

    df = df[df['number_of_vehicles'] <= 6]

    features = [
        'location_easting_osgr', 'location_northing_osgr', 'speed_limit', 'light_conditions', 'weather_conditions', 'road_surface_conditions', 'hour', 'day_of_week'
    ]

    X = df[features]
    y = np.log1p(df['number_of_vehicles'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingRegressor(
        n_estimators=300, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test)

    print("\n[Analysis Results]")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test_real, y_pred):.3f} vehicles")
    print(f"R-Squared Score: {r2_score(y_test_real, y_pred):.4f}")

    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_test_real, y=y_pred, scatter_kws={'alpha':0.2}, line_kws={'color':'red'})
    plt.title('Sheffield Regression Analysis: Predicted vs Actual Vehicle Counts')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('regression_performance.png')
    print("Plot saved: regression_performance.png")

if __name__ == "__main__":
    run_regression_analysis()