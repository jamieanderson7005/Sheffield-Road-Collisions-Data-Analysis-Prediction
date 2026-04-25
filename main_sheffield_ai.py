import data_cleaning
import regression_analysis
import supervised_learning
import unsupervised_learning
import comprehensive_workflow
import os

def main():
    if not os.path.exists('Results'):
        os.makedirs('Results')
        print("Created 'Results' folder for visualisations.")

    print("Master entry point for the Sheffield Road Collision Analysis")

    print("Running data cleaning and pre-processing")
    data_cleaning.clean_sheffield_data("Filtered_Sheffield_Traffic_Data.csv")
    print("'cleaned_data.csv is ready")

    print("Running Binary Classification")
    comprehensive_workflow.run_comprehensive_workflow()

    print("Running Multiclass Severity Analysis")
    supervised_learning.run_supervised_learning()

    print("Running Regression Prediction")
    regression_analysis.run_regression_analysis()

    print("Running Unsupervised Clustering")
    unsupervised_learning.run_unsupervised_learning()

if __name__ == "__main__":
    main()
    