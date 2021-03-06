# IBM HR Attrition and Performance Prediction  

## Collaborators

* Megan Madrigal
* Johan Bastos
* Cesar Orozco

## Brief Description

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

With RandomForest MontlyIncome, OverTime and Age are the most important. Because a common ground is always important, the following plot represents the two features that weight more for model calculations:


<p align="center">
  <img src="img/most_import_features.png" />
</p>

Interestingly, with Age, people between 25 to 35 seems to be prone to leave. There is a sligth peak during the 50's, but general speaking young adults has the tendency to change jobs or leave. 

Income is the most important over all factors to predict if a person leave the company or not. The population that earns less than $2500 per month tend to leave more frequently. 

Something interesting to noticied is how AdaBoost get rid of MontlyYear, Gender, Education, HourlyRate, PerormanceRating whereas RandomForest give to some of them a relative high weigth such as MontlyRate feature. 


In summary, there is no surprise that people who earns less tend to look for other opportunities or improvements to have a better position, also it is not a surprise that people between 25 to 35 is more suceptible for leaving.   


## Improvements

Although the `Adaboost` brings the best for all results, there is room to improve to score. The first improvement is to test more hyperparameters such as:

base_estimator: It is a weak learner used to train the model. It uses DecisionTreeClassifier as default weak learner for training purpose. You can also specify different machine learning algorithms.
n_estimators: Number of weak learners to train iteratively.
learning_rate: It contributes to the weights of weak learners. It uses 1 as a default value.


There are also some pros and cons with respect the use of Adaboost:

Pros
* AdaBoost is easy to implement. It iteratively corrects the mistakes of the weak classifier and improves accuracy by combining weak learners. You can use many base classifiers with AdaBoost. 
* AdaBoost is not prone to overfitting. This can be found out via experiment results, but there is no concrete reason available.

Cons
* AdaBoost is sensitive to noise data. It is highly affected by outliers because it tries to fit each point perfectly. AdaBoost is slower compared to XGBoost.

So, given the Cons, what is needed in this dataset is removing outliers. 

<p align="center">
  <img src="img/outliers.png" />
</p>

The features that have outliers are `MonthlyIncome, Age, YearsAtCompany, TotalWorkingYears, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager`. 

Another approach to for improvements is to try different paramenters with every single model using the `insights` function above. 


In summary, to predict accuratetly is not just matter of running a model, as stated frequently, we need to know our information or in many cases consult with an expert. Then, it is possible to take a decision on how to treat information, whether to remove outliers, scale data, etc. 
