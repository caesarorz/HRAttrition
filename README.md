# IBM HR Attrition and Performance Prediction  



## Description

Group Name: Grumpy Grinches

URL: `https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset` 


The following a is the IBM Human Resources Attrition analisys and prediction. The main goal is to predict the employees's attrition using different models.

Below the results for the EDA and prediction.

## Results


### Dataset Manipulation

At first glance this is a classification problem. The dataset contains 35 columns and 1470 observations:

```
df.shape
(1470, 35)
```

The next step is to check out what kind of data is. A simply solution is to take a look for unique values (if they exist):

```
for i in df:
    print('*************************************************************')
    print('Column name', i, '      column type', df[i].dtypes)
    print(df[i].unique())
```

Columns that do not have any meaning are removed due to the fact that, for instance, `EmployeeCount` just have one value for every single observation (1 as a value). The following function was used to remove them in a shot, at the same time ignore errors if the notebook is run again.  

```
# drop unneccesary columns: EmployeeNumber, DailyRate
columns_to_drop = ['DailyRate', 'EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']

def removeColumns(columns):
    for i in columns:
        if i in df:
            df.drop(i, inplace=True, axis=1)

removeColumns(columns_to_drop)
```

Once the dataset is cleaner, althoght there are many steps in between (like cleaning and analyzing), the separation for feature and target is the crucial part:

```
X = df.drop('Attrition', axis=1)
y = df['Attrition']

def onehot():
    """Encode categorical and combine categorical and numerical in one df"""
    new_list = [df.select_dtypes(['int64'])]
    for i in X.select_dtypes(['object']).columns:
        temp = pd.get_dummies(X[i], prefix=i)
        new_list.append(temp)
    return pd.concat(new_list, axis=1)
    
df_ = onehot()

X_train, X_test, y_train, y_test = train_test_split(df_, y)
```

As we noticed in the code above, the use of `onehot` function along with the splitting is present. 

### Usage of models

The use of models in this case is very conservative. It start with `DecisionTreeClassifier` without any hyperparameter, then `RandomForest`, `Bagging` and others. Nevertheless, the project contains a summary for all different models. The code below, is a custom way that summarize all results: 

```
#log_reg_params = [{"C":0.01}, {"C":0.1}, {"C":1}, {"C":10}, {"max_iter": 10000}]
dec_tree_params = [{"criterion": "gini"}, {"criterion": "entropy"}]
rand_for_params = [{"criterion": "gini"}, {"criterion": "entropy"}, {"min_samples_leaf": 50}]  
kneighbors_params = [{"n_neighbors":3}, {"n_neighbors":5}]
naive_bayes_params = [{}]
svc_params = [{"C":0.01}, {"C":0.1}, {"C":1}, {"C":10}]
ada_params = [{}]
bag_params = [{}]
gra_params = [{}]

modelclasses = [
    # ["log regression", LogisticRegression, log_reg_params],
    ["decision tree", DecisionTreeClassifier, dec_tree_params],
    ["random forest", RandomForestClassifier, rand_for_params],
    ["k neighbors", KNeighborsClassifier, kneighbors_params],
    ["naive bayes", GaussianNB, naive_bayes_params],
    ["support vector machines", SVC, svc_params],
    ['ada boost classifier', AdaBoostClassifier, ada_params],
    ['bagging classifier', BaggingClassifier, bag_params],
    ['gradient classifier', GradientBoostingClassifier, gra_params],
]

def insights(X_train, y_train, X_test, y_test):
    insights = []
    for modelname, Model, params_list in modelclasses:
        for params in params_list:
            model = Model(**params)
            model.fit(X_train, y_train.ravel())
            score = model.score(X_test, y_test.ravel())
            insights.append((modelname, model, params, score))

    return insights

results = insights(X_train, y_train, X_test, y_test)
```

<p align="center">
  <img src="img/model_results.PNG" />
</p>

`AdaBoostClassifier` got the best of all results. Let's analyze a little bit more AdaBoost. 

To get a better graps with Ada, lets get accuracy_score, classification_report, confusion_matrix from sklearn.metrics:


```
***************************** Training ****************************************************
Accuracy:		 0.9038112522686026
Classification Report:
               precision    recall  f1-score   support

           0       0.91      0.98      0.95       928
           1       0.83      0.49      0.62       174

    accuracy                           0.90      1102
   macro avg       0.87      0.74      0.78      1102
weighted avg       0.90      0.90      0.89      1102

Confusion Matrix:
 [[911  17]
 [ 89  85]]
******Cross Validation Number: 10
Avg accuracy: 0.8784193284193282
Accuracy standard dev: 0.02470379229026582

***************************** Testing ****************************************************
Accuracy:		 0.875
Classification Report:
               precision    recall  f1-score   support

           0       0.88      0.98      0.93       305
           1       0.77      0.38      0.51        63

    accuracy                           0.88       368
   macro avg       0.83      0.68      0.72       368
weighted avg       0.87      0.88      0.86       368

Confusion Matrix:
 [[298   7]
 [ 39  24]]

 ```

Note: The function is inside the notebook.

For training the accuracy is 0.90 and 0.87 for testing. For all models this is the best result. Average accuracy is 0.878 with standard deviation of 0.02 rounded, which is pretty low.

### Most important features

Using the `best_estimators_` value from Adaboost, the following plot is rendered:

<p align="center">
  <img src="img/ada_best_estimator.png" />
</p>

As can be seen, the three most important estimators for the model are MontlyIncome, Age, and YearsWithCurrManager. 

In another point of view (from RandomForest Class) the following result is rendered:


<p align="center">
  <img src="img/random_best_estimators.png" />
</p>

With RandomForest MontlyIncome, OverTime and Age are the most important. 

