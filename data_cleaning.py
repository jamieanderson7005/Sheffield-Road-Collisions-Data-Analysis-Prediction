import pandas as pd
import numpy as np

def clean_sheffield_data(path):
    print("Beginning Data Pre_Processing")
    df = pd.read_csv(path, low_memory=False)
    
    #Identifying useful columns
    cols_to_keep = [
        'collision_severity', 'number_of_vehicles', 'day_of_week', 'speed_limit', 'light_conditions', 'weather_conditions', 'road_surface_conditions', 'urban_or_rural_area', 'location_easting_osgr', 'location_northing_osgr', 'time'
    ]
    df_filtered = df[cols_to_keep].copy()

    #Handling missing values
    df_filtered.dropna(inplace=True)

    df_filtered['hour'] = df_filtered['time'].str.split(':').str[0].astype(int)
    df_filtered.drop(columns=['time'], inplace=True)
    
    print(f"Dataset has been cleaned, rows remaining after cleaning: {len(df_filtered)}")
    return df_filtered

if __name__ == "__main__":
    data = clean_sheffield_data('Filtered_Sheffield_Traffic_Data.csv')
    data.to_csv('cleaned_data.csv', index=False)
    print("Saved Data to cleaned_data.csv")
