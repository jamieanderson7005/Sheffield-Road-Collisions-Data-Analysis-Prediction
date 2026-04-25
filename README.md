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