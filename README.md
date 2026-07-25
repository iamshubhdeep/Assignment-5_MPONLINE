# Employee Attrition Prediction using Decision Tree and Random Forest

## Details

- Name : Shubhdeep Singh
- Application No. : IN26011804
- College Reg. No. : 23BCE11460
- Batch No. : 1(A)
- Email Id : shubhdeep.23bce11460@vitbhopal.ac.in

## Objective
Predict whether an employee is likely to leave the organization (attrition) based on
demographic, professional, and work-related attributes, and compare the performance of a
Decision Tree Classifier against a Random Forest Classifier.

## Dataset
IBM HR Analytics Employee Attrition & Performance Dataset
Kaggle link: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset


## Libraries Used
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- kagglehub (optional, for automatic dataset download)

Install with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn kagglehub
```

## Methodology
1. **Data Understanding** – Loaded the dataset, inspected the first five records, identified
   numerical vs. categorical features and the target variable (`Attrition`), and reviewed
   dataset info/summary statistics.
2. **Data Preprocessing** – Checked for missing values, dropped uninformative columns
   (`EmployeeCount`, `StandardHours`, `Over18`, `EmployeeNumber`), label-encoded categorical
   variables, and split the data into 80% training / 20% testing sets (stratified on the
   target).
3. **Model Development** – Trained a `DecisionTreeClassifier` and a `RandomForestClassifier`
   (`n_estimators=100`) on the same training data.
4. **Model Evaluation** – Compared both models using Accuracy, Precision, Recall, and F1-Score,
   generated confusion matrices for each, and plotted feature importances from the Random
   Forest model.
5. **Bonus (optional)** – Tuned `max_depth` for the Decision Tree across several values and
   reported the effect on performance.

## Results

| Model          |  Accuracy  | Precision | Recall | F1-Score |
|----------------|------------|-----------|--------|----------|
| Decision Tree  |    0.79    |    0.42   |  0.38  |   0.40   |
| Random Forest  |    0.87    |    0.65   |  0.31  |   0.42   |

Confusion matrices: see `confusion_matrices.png`
Feature importance plot: see `feature_importance.png`

## Model Comparison

- Random Forest generally achieves higher accuracy and a better F1-score than the single
  Decision Tree, since averaging across many trees reduces variance and overfitting.
- The Decision Tree tends to overfit the training data more, which can show up as a larger
  gap between training and test performance.
- The confusion matrices reveal how many actual "Yes" (attrition) cases each model catches
  vs. misses — this is often the more important error to minimize in an HR context.
- The Random Forest feature importance plot highlights the most influential drivers of
  attrition (e.g., OverTime, MonthlyIncome, Age, TotalWorkingYears), which can guide HR
  retention strategy.

## Conclusion


Both a Decision Tree and a Random Forest classifier were trained to predict employee attrition.
Based on the evaluation metrics, [state which model performed better]. Random Forest often
outperforms a single Decision Tree because it is an ensemble (bagging) method: it trains many
trees on bootstrapped samples with random feature subsets and averages their predictions,
which reduces variance and overfitting compared to any one tree. A key limitation of Decision
Trees is that they are highly sensitive to small changes in the data and prone to overfitting
without depth limits or pruning. A key limitation of Random Forests is reduced interpretability
— it's harder to trace a single prediction to a clear decision path — along with higher
computational cost for training and inference compared to a single tree.

## Repository Structure
```
.
├── Assignment-5.ipynb   
├── README.md
├── confusion_matrices.png   
└── feature_importance.png   
```
