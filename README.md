# Sheffield Road Collisions Data Analysis Prediction
- Jamie Anderson
- Student ID: c4021414

---

## Project Description
For me, this project was all about developing a data driven framework that analyses road safety in Sheffield. It does this by using supervised/unsupervised learning and regression models, the system identifies patterns in accident severity and claasifies urban vs rural collison zones. It will the predict how many cehicles were involved using the data set.

## File Organisation
- **data_cleaning.py:** This is the entry point to my program, it processes the raw data provided to us from the data set ,   filters for the Sheffield city code (215) and generates the file: **cleaned_data.csv**.
- **comprehensive_workflow:** This file performs the binary classification (urban vs rural) and includes the significance testing.
- **supervised_learning.py:** This file performs the multiclass classification (slight, serious or fatal), it also uses SMOTE for class balancing to improve the accuracy score.
- **unsupervised_learning.py:** This file is responsible for making the **accident_hotspots.png** graph as it displays where the crashes took place with northing and easting coordinates.
- **regression_analysis.py:** This file performs numerical predictions for the number of vehicles involved with the accident.
- **main_sheffield_ai.py:** This file is the master script as it will execute all of the files in the correct sequence when it has been ran.
- **Results:** This is the folder that includes all of the graphs used to visualise the data (Confusion matrices, scatter graphs, etc).

## Dependencies:
1. pandas
2. scikit-learn
3. imbalanced-learn
4. matplotlib
5. seaborn
6. numpy

## Installation Instructions
1. Extract all the files: Right click the compressed file (c4021414_Jamie.zip) and select Extract All
2. Esnure you have python installed
3. Install required tools:

`pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn`

## Usage
1. Ensure the raw dataset is in the directory
2. Run the master script:

`python main_sheffield_ai.py`

Note: The script will automatically verify the data, create the Results folder, and generate all charts/graphs.

## AI Transparency Statement Declaration
# Level 2: AI for Shaping
AI was used in this project during the inital stages of the activity to help outline the system architecture and suggesting performance metrics. All modelling and logic were written and refined by myself.

# AI prompt Logs
1. Outlining: "Help me outline a Python script structure to run multiple ML models in an automated pipeline"
2. Refining: "Which specific performance metrics are best for evaluating regression models predicting numerical counts?"