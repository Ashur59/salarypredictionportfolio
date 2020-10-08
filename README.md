
# Salary Prediction Project (Python)

## Assignment Description

Examining a set of job postings with salaries and then predict salaries for a new set of job postings.

## Data Supplied

I was given three CSV data files:

* ```train_features.csv```:

    Each row represents metadata for an individual job posting. The "jobId" column represents a unique identifier for the job posting. The remaining columns describe features of the job posting.
* ```train_salaries.csv```:

    Each row associates a "jobId" with a "salary".
* ```test_features.csv```:

    Similar to ```train_features.csv```, each row represents metadata for an individual job posting.

    The first row of each file contains headers for the columns. Keep in mind that the metadata and salary data may contain errors.

## The task

I must build a model to predict the salaries for the job postings contained in ```test_features.csv```. The output of my system should be a CSV file entitled ```test_salaries.csv``` where each row has the following format:

* jobId, salary

As a reference, my output should mirror the format of ```train_salaries.csv```.

## Methodology

### Introduction

To achieve the goal mentioned above, this notebook is written to train a machine learning algorithm on the salary "train" dataset and, then, to score the "test" dataset. Given the known salary for a big set of Job ID listings along with some known (relevant) features, i.e. Company ID, Job Type, Degree, Major, Industry, Years Experience, and Distance from Metropolitan, I developed a machine learned statistical model to predict the salary for another dataset with the same exact features as the one having known target "salary". In other words, I trained my model based on the "train" dataset with known salaries and, then, tested it on the "test" dataset with unknown salaries. This is a supervised machine learning problem which is relevant for lots of applications, i.e. glassdoor, in order to predict salary for any given Job ID, e.g. position.

### Software Language and Libraries Utilized

To solve the problem, I used **_python3_** and its libraries, i.e. **NumPy**, **SciPy**, **Pandas**, **xgboost**, and **pdpbox** as well as **scikit-learn**, and visualization tools, **matplotlib** and **Seaborn**.

### Data Preparation (Cleansing)

First, I converted features from *Object* to *Category* type. Then, I removed very few rows in the data which were missing target variable, "salary" upon which all training is supposed to take place. There was no ```NAN``` or missing data. I converted all categorical features into numerical features using *_onehot_* and *_label_* encoding. Specifically, "industry," "major," and "companyId" were *non-ordinal* (and hence onehot encoded) while "degree" and "jobType" were considered *_ordinal_* (and hence label encoded).  Since all \~1,000,000 "jobId"s are unique and that "companyId" is irrelevant in terms of salary prediction, I removed them both. I converted long ```Int64``` datatypes into short ```Int8``` and ```Int16``` to reduce memory use. Given that the goal of the problem is to investigate the salary for *entry-level* vs. *senior-level* data science roles, I created 4 categoricals depending on numerical values of "yearsExperience" feature. [0, 2] was considered entry-level while [5, 9] as senior level.

### Algorithmic Methods Utilized

The Algorithms I considered were, **_Lasso_**, **_RandomForestRegressor_**, **_DecisionTreeRegressor_**,**_LinearRegressor_**, **_Ridge_**, **_GradientBoostingRegressor_**, **_XGBRegressor_**, and **_KNeighborsRegressor_**. After spot testing each model with ```Cross-validation``` evaluating each model in turn, I chose the three highest-score algorithms:

* **_Ridge_**
* **_GradientBoostingRegressor_**
* **_XGBRegressor_**

These three algorithms produced the least *mean squared error*, (*MSE*) as our metric of the problem. After tuning hyper parameters of the 3 best algorithms, I chose **_XGBRegressor_** which had the lowest MSE for both *entry-level* and *senior-level* data science roles.

### The Way **_XGBRegressor_** Works

The reason **_XGBRegressor_** is preferred over normal **_GradientBoostingRegressor_** is having to deal with a large-scale "train" dataset with \~1,000,000 unique data points whose features were mostly categoricals. The **xgboost** python package is faster than **scikit-learn** implementation of normal gradient boosting. **xgboost** is an ensemble method that combines many decision trees to create a strong model out of many weak learners by building trees in a serial manner, where each individual tree corrects the error of the previous one. The utilized shallow trees make it smaller in memory and faster in prediction. Due to the absence of any randomization mechanism, the model is sensitive to hyper parameters. Specifically, the most important hyper parameters are:

* ```"learning-rate"```:

    It controls the strength of each tree in correcting the errors of previous ones; for complex models, a higher “rate” is advised.
* ```"n_estimators"```:

    It is the number of trees that controls the same strength; higher number corresponds to more complex problems.

The model would have worked better if the data was not as sparse as it is (after _feature engineering_) given considerable number of categorical features.

### The Features Utilized

I used all features in the original "train" data except "jobId" and "companyId" as they were completely irrelevant after investigating the importance of features.

### Training the Model

Instead of implementing a simple _grid_ search over the parameters of each model, training and evaluating a regressor for each combination to assess how good the model is, and to avoid information _leaking,_ I split the data into three sets: the _training_ set to build the model, the _validation_ set to select the parameters of the model, and the _test_ set to evaluate the performance of the selected parameters. Specifically, I trained **_XGBRegressor_** algorithm on 80% of "train" data in order to score the 20% "validation" data. For a better estimate of the generalization performance, I used _cross-validation_ to evaluate the performance of each parameter combination using the **_GridSearchCV_** class which implements the methodology in the form of an estimator. Fitting the **_GridSearchCV_** object both searches for the best parameters, and automatically fits a new model on the whole training dataset with the parameters that yield the best _cross-validation_ performance. With the best hyper-parameters:

* ```booster='gbtree'```
* ```gamma=0```  
* ```learning_rate=0.2```
* ```max_depth=6```
* ```n_estimators=100```
* ```reg_alpha=0```
* ```reg_lambda=1```
* ```tree_method='exact'```,

I made sure that I am hitting the required _MSE_ for both _entry-level_ and _senior-level_ data science roles among all training dataset by training the same model over the entire "train" dataset; _entry-level_ required \<= 360 and _senior-leve_ required \<= 320. While it is not difficult to hit the required _accuracy_ on the 4 distinct subsets of "train" dataset, i.e. entry, junior, senior, and principal, it was challenging to get the same high _accuracy_ on the entire dataset all at once where "yearsExperience" feature is encoded into 4 _ordinal_ distinct values.

### Assessing the Accuracy of the Predictions

While for any _regression_ model, we have a plethora of evaluation metrics, e.g. _mean squared error (MSE)_, _root mean squared error (RMSE)_, _mean absolute error (MAE)_, and _coefficient of determination (R^2),_ I considered _mean squared error (MSE)_ for assessing the accuracy of my prediction as requested by the business problem. While in general *_R^2_* is a more intuitive metric to evaluate _regression_ models, the business decisions in this problem was made based on _MSE_.

### Feature Importance

In a descending order, "industry," "major," "jobType," "yearsExperience," and "degree" affected the salary predictions the most. "milesFromMetropolis" and "companyId" had the least impact on the salary prediction. I identified this by applying the ```feature_importances_``` method on the chosen machine learning algorithm after fitting on the final _feature-engineered_ and _combined_ "train" data. _partial dependence plots_, on the other hand, showed *_how_* each one of the features affected the salary predictions.

### Additional Works

For the training dataset, I made visualizations of salary distribution in the form of _boxplots_ (using **_Seaborn_** python package) marginalized over any one of the features. I also plotted the correlation matrix among _numerical_ and _ordinal_ categorical features in order to find any mutual correlation between any two of such features to justify the previously used and engineered features. At last, I included the _partial dependence plots_ from both **scikit-learn** and **python**.
