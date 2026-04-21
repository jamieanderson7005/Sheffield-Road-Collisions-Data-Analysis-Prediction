import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

def run_supervised_learning():
    print("Starting Supervised Learning Task")

    df = pd.read_csv('cleaned_data.csv')

    x = df.drop(columns=['collision_severity', 'location_easting_osgr', 'location_northing_osgr'])
    y = df['collision_severity']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    print("Training tuned random forrest")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

    rf_model.fit(x_train_scaled, y_train)
    rf_preds = rf_model.predict(x_test_scaled)

    acc = accuracy_score(y_test, rf_preds)
    print(f"Random forest accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, rf_preds))

if __name__ == "__main__":
    run_supervised_learning()