import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def run_comprehensive_workflow():
    print("Initializing Significant Sheffield AI Framework")

    try:
        df = pd.read_csv('cleaned_data.csv')
    except FileNotFoundError:
        print("Error: cleaned_data.csv not found")
        return

    df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if x in [8, 9, 16, 17] else 0)

    df = df[df['urban_or_rural_area'].isin([1, 2])]
    print(f"Dataset filtered. Rush hour features engineered. Rows: {len(df)}")

    features = [
        'location_northing_osgr', 'location_easting_osgr',
        'speed_limit', 'light_conditions', 'weather_conditions',
        'hour', 'is_rush_hour' 
    ]

    X = df[features]
    y = df['urban_or_rural_area']

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("\n[Analysis] Investigating Model Complexity (n_estimators)...")
    configs = [50, 200]
    for n in configs:
        temp_model = RandomForestClassifier(n_estimators=n, class_weight='balanced', random_state=42)
        temp_model.fit(X_train_scaled, y_train)
        score = accuracy_score(y_val, temp_model.predict(X_val_scaled))
        print(f"Testing {n} trees: Validation Accuracy = {score*100:.2f}%")

    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    model.fit(X_train_scaled, y_train)

    # 4. EVALUATION & VISUALIZATION
    print("\n[Final Results] Evaluating on Unseen Test Data:")
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Urban', 'Rural'], yticklabels=['Urban', 'Rural'])
    plt.title('Sheffield Accident Classification: Confusion Matrix')
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.savefig('Results/performance_evaluation_matrix.png')

    print("\nResponsible AI & Ethics: ")
    print(f"Model utilizes {len(features)} features for geographic prediction.")
    print("Sustainability: Balanced class weights used to prevent Rural data neglect.")

if __name__ == "__main__":
    run_comprehensive_workflow()