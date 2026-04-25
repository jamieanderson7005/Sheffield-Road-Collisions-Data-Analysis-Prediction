import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

def run_supervised_learning():
    print("Starting Supervised Learning Task")

    df = pd.read_csv('cleaned_data.csv')
    
    severity_map = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
    df['severity_label'] = df['collision_severity'].map(severity_map)

    features = ['weather_conditions', 'light_conditions', 'speed_limit', 'number_of_vehicles', 'road_surface_conditions', 'hour', 'location_easting_osgr', 'location_northing_osgr']

    X = df[features]
    y = df['severity_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train_scaled, y_train)

    print("Severity Classification Results: ")
    y_pred = model.predict(X_test_scaled)

    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    run_supervised_learning()