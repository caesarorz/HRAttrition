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

## Usage of models

The use of models in this case is very conservative. It start with `DecisionTreeClassifier` without any hyperparameter, then `RandomForest`, `Bagging` and others. Nevertheless, the project contains a summary for all 

