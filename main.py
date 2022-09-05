import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


class RemoveFeatures(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Transformer to remove unnecessary
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:, 6:8]

# Reading csv files
df_train = pd.read_csv('internship_train.csv')
X_train = df_train.drop(['target'], axis=1).to_numpy()
y_train = df_train['target']

test_df = pd.read_csv('internship_hidden_test.csv')
X_test = test_df.to_numpy()

# Creating pipeline for X's
pipe = Pipeline(steps=[
    ('remove_features', RemoveFeatures()),
    ('scale', StandardScaler()),
    ('polynomial', PolynomialFeatures(degree=2, include_bias=False))
])

X_train = pipe.fit_transform(X_train)
X_test = pipe.fit_transform(X_test)

# Training model
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predicting test set
y_test_preds = regression.predict(X_test)

# Saving predictions to test dataframe
test_df['target'] = y_test_preds

# Saving dataframe with predictions to a new file
test_df.to_csv('internship_test.csv', index=False)




