# Supervised Machine Learning Models

## Project Overview
This project implements and compares two supervised machine learning regression algorithms using Scikit-learn.

## Objective
To understand model training, prediction, and evaluation using performance metrics.

## Algorithms Implemented
1. Linear Regression
2. Decision Tree Regression

## Dataset
A sample dataset containing study hours and corresponding student marks is created using Pandas.

- **Feature:** Study Hours
- **Target:** Marks

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn

## Workflow
1. Create the dataset.
2. Separate features and target.
3. Split the data into training and testing sets.
4. Train both regression models.
5. Generate predictions.
6. Evaluate models using MAE, MSE, and R² Score.
7. Compare model performance and visualize the data.

## Evaluation Metrics
- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **R² Score:** Measures how well the model explains variation in the target.

## How to Run
Install the required libraries:

```bash
pip install pandas matplotlib scikit-learn
```

Run the project:

```bash
python supervised_ml.py
```

## Output
The program prints evaluation metrics and a comparison table. It also saves `model_comparison.csv` and `study_hours_vs_marks.png`.

## Conclusion
This project demonstrates how two supervised regression models can be trained and compared using evaluation metrics. The results help illustrate their performance on the sample dataset.