import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('./data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
features = tpot_data.drop('Attrition', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Attrition'], random_state=42)

# Average CV score on the training set was: 0.8920311220311221
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=1, min_samples_leaf=6, min_samples_split=4)),
    StackingEstimator(estimator=GaussianNB()),
    MinMaxScaler(),
    StackingEstimator(estimator=LinearSVC(C=5.0, dual=True, loss="hinge", penalty="l2", tol=0.001)),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=1, max_features=0.45, min_samples_leaf=1, min_samples_split=16, n_estimators=100, subsample=0.6500000000000001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
