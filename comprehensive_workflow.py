import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def run_comprehensive_workflow():
    print("Starting comprehensive workflow")

    try:
        df = pd.read_csv('cleaned_data.csv')
        print(df.columns.tolist())
    except FileNotFoundError:
        print("Error: cleaned_data.csv not found")
        return
    
    df = df[df['urban_or_rural_area'].isin([1, 2])]
    print(f"Filtered dataset to valid Urban/Rural classes. Rows remaining: {len(df)}")
    
    print("\nBinary Classification: Urban vs Rural")

    features = [
        'location_northing_osgr',
        'location_easting_osgr',
        'speed_limit',
        'light_conditions',
        'weather_conditions',
        'hour',
    ]

    X = df[features]
    y = df['urban_or_rural_area']

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("Data successfully scaled using StandardScaler")

    model = RandomForestClassifier(n_estimators=200,class_weight='balanced', random_state=42)
    model.fit(X_train_scaled, y_train)

    val_preds = model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Current Validation Accuracy: {val_acc*100:.2f}%")

    print("\nChecking for Overfitting with Cross-Validation")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=kf)
    print(f"Mean CV Score: {np.mean(cv_scores)*100:.2f}%")

    print("Evaluation on Unseen Test Data: ")
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Sheffield Accident Classification: Confusion Matrix')
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.savefig('performance_evaluation_matrix.png')
    print("Plot saved")

    print("\nResponsible AI")

    class_counts = df['urban_or_rural_area'].value_counts(normalize=True)
    print("Class Distribution Obeserved: ")
    print(class_counts)

    if class_counts.max() >0.8:
        print("High class inmbalance detected")
    else:
        print("Data distribution balanced")
    
if __name__ == "__main__":
    run_comprehensive_workflow()