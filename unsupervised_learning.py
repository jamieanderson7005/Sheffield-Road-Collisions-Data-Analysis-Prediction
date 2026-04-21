import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

def run_unsupervised_learning():
    print("Starting unsupervised learning")
    df = pd.read_csv('cleaned_data.csv')

    df = df[
        (df['location_easting_osgr'] > 420000) & (df['location_easting_osgr'] < 450000) &
        (df['location_northing_osgr'] > 370000) & (df['location_northing_osgr'] < 400000)
    ]

    if df.empty:
        print("Error, filters too tight")
        return

    coords = df[['location_easting_osgr', 'location_northing_osgr']]

    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
    df['accident_cluster'] = kmeans.fit_predict(coords)

    plt.figure(figsize=(10,8))
    sns.scatterplot(
        data=df,
        x='location_easting_osgr',
        y='location_northing_osgr',
        hue='accident_cluster',
        palette='bright',
        alpha=0.7
    )

    plt.title('Unsupervised Learning: Accident Hotspots')
    plt.xlabel('Easting')
    plt.ylabel('Northing')

    plt.savefig('accident_hotspots.png')
    print("Hotspot Map Saved")

if __name__ == "__main__":
    run_unsupervised_learning()