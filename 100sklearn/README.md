# Sklearn

- [Sklearn](#sklearn)
  - [Choosing the right Estimator](#choosing-the-right-estimator)
    - [Regression](#regression)
      - [Ensemble methods](#ensemble-methods)
    - [Classification](#classification)
      - [Ensemble methods for classification](#ensemble-methods-for-classification)
  - [Making Predictions](#making-predictions)
    - [With Classification Model](#with-classification-model)
      - [`predict()`](#predict)
      - [`predict_proba()`](#predict_proba)
    - [With Regression Model](#with-regression-model)
  - [Evaluation](#evaluation)
    - [`model.score(X_test, y_test)`](#modelscorex_test-y_test)
    - [cross validation with `cross_val_score()`](#cross-validation-with-cross_val_score)
    - [Problem Specific Metrics for Classification](#problem-specific-metrics-for-classification)
      - [Accuracy](#accuracy)
      - [Area Under the ROC Curve](#area-under-the-roc-curve)
  - [Putting All Together: PipeLine + GridSearchCV](#putting-all-together-pipeline--gridsearchcv)
    - [v1: Pipelining up to Single Estimator](#v1-pipelining-up-to-single-estimator)
      - [1. EDA](#1-eda)
      - [2. Define numerical and categorical features.](#2-define-numerical-and-categorical-features)
      - [3. PipeLine 1: For categorical features: fill missing values then label encode](#3-pipeline-1-for-categorical-features-fill-missing-values-then-label-encode)
      - [4. PipeLine 2: For numerical features: fill missing values then feature scale](#4-pipeline-2-for-numerical-features-fill-missing-values-then-feature-scale)
      - [5. Combine numerical and categorical transformer using `ColumnTransformer`.](#5-combine-numerical-and-categorical-transformer-using-columntransformer)
      - [6. (optional) Apply PCA to reduce dimensions.](#6-optional-apply-pca-to-reduce-dimensions)
      - [7. Final Pipeline With an Estimator](#7-final-pipeline-with-an-estimator)
      - [Visualizing Pipeline](#visualizing-pipeline)
  - [Resource](#resource)

```python
"""
cd .\100sklearn\
jupyter nbconvert --to markdown sklearn.ipynb --output README.md
"""
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('jpg')
```

## Choosing the right Estimator


- [https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)

<div align="center">
<img src="img/es.jpg" alt="es.jpg" width="900px">
</div>

### Regression


```python
# from sklearn.datasets import load_boston
# boston = load_boston()
# boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
# boston_df['target'] = boston.target
# boston_df.head()
# !FutureWarning: load_boston is deprecated; `load_boston` is deprecated in 1.0 and will be removed in 1.2.
```


```python
df = pd.read_csv("Boston.csv",index_col=0)
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>crim</th>
      <th>zn</th>
      <th>indus</th>
      <th>chas</th>
      <th>nox</th>
      <th>rm</th>
      <th>age</th>
      <th>dis</th>
      <th>rad</th>
      <th>tax</th>
      <th>ptratio</th>
      <th>black</th>
      <th>lstat</th>
      <th>medv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1</td>
      <td>296</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (506, 14)




```python
from sklearn.model_selection import train_test_split
# let's try the ridge regression model
from sklearn.linear_model import Ridge
# random seed for reproducibility
np.random.seed(0)

# create the data
X = df.drop('medv', axis=1)
y = df['medv']

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# instantiate the model
model = Ridge()

# fit the model (taring the machine learning model)
model.fit(X_train, y_train)

# check the score on the test set (use the patterns the model has learned)
model.score(X_test, y_test)
```

    (404, 13) (102, 13) (404,) (102,)





    0.5796111714164921



 - How do we improve the model?
 - What if Ridge regression wasn't working?

#### Ensemble methods

The goal of ensemble methods is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability / robustness over a single estimator.

- [https://scikit-learn.org/stable/modules/ensemble.html](https://scikit-learn.org/stable/modules/ensemble.html)


```python
from sklearn.model_selection import train_test_split
# let try the random forest regressor
from sklearn.ensemble import RandomForestRegressor

# seed the random number generator
np.random.seed(0)

# create the data
X = df.drop('medv', axis=1)
y = df['medv']

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# instantiate and fit the model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# check the score on the test set
model.score(X_test, y_test)
```

    (404, 13) (102, 13) (404,) (102,)





    0.7734908201180223



### Classification


```python
df = pd.read_csv("heart-disease.csv")
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>1</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>1</td>
      <td>1</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>1</td>
      <td>178</td>
      <td>0</td>
      <td>0.8</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>354</td>
      <td>0</td>
      <td>1</td>
      <td>163</td>
      <td>1</td>
      <td>0.6</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Consulting the estimator map above, let's try `LinearSVC`


```python
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC

# seed the random number generator
np.random.seed(0)

# create the data
X = df.drop('target', axis=1)
y = df['target']

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# instantiate the model
model = LinearSVC()
model.fit(X_train, y_train)

# check the score on the test set
model.score(X_test, y_test)
```

    (242, 13) (61, 13) (242,) (61,)


    c:\Users\soiko\anaconda3\lib\site-packages\sklearn\svm\_base.py:1206: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      warnings.warn(





    0.45901639344262296



#### Ensemble methods for classification


```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# seed the random number generator
np.random.seed(0)

# create the data
X = df.drop('target', axis=1)
y = df['target']

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# instantiate the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# check the score on the test set
model.score(X_test, y_test)
```

    (242, 13) (61, 13) (242,) (61,)





    0.8852459016393442



Titbit:

- If you have structured data, use `ensemble` methods
- If you have unstructured data, use `deep learning` or `transfer learning`

## Making Predictions

### With Classification Model

Using
- `predict()`
- `predict_proba()`


```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("heart-disease.csv")
np.random.seed(0)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# instantiate the model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# check the score on the test set
clf.score(X_test, y_test)
```

    (242, 13) (61, 13) (242,) (61,)





    0.8852459016393442



#### `predict()`


```python
# get prediction:
clf.predict(X_test)

```




    array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0,
           0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0,
           0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=int64)




```python
# original label
np.array(y_test)
```




    array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,
           0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0,
           1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1], dtype=int64)




```python
# compare the prediction with the truth label to evaluate the model
y_preds = clf.predict(X_test)
np.mean(y_preds == y_test)
```




    0.8852459016393442



using `accuracy_score(y_test, y_preds)`:


```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_preds)
```




    0.8852459016393442



which is same as the `score(X_test, y_test)` method in scikit-learn


```python
# ~
clf.score(X_test, y_test)

```




    0.8852459016393442



#### `predict_proba()`


```python
# return probabilities of a classification model
clf.predict_proba(X_test[:5])

```




    array([[0.9 , 0.1 ],
           [0.56, 0.44],
           [0.5 , 0.5 ],
           [1.  , 0.  ],
           [0.88, 0.12]])



### With Regression Model


```python
df = pd.read_csv("Boston.csv",index_col=0)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
np.random.seed(0)
X = df.drop('medv', axis=1)
y = df['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
model.score(X_test, y_test)
```

    (404, 13) (102, 13) (404,) (102,)





    0.7734908201180223




```python
y_preds = model.predict(X_test)
y_preds[:5]
```




    array([23.932, 29.111, 22.14 , 11.006, 20.543])




```python
np.array(y_test[:5])

```




    array([22.6, 50. , 23. ,  8.3, 21.2])




```python
# compare the prediction with the truth label to evaluate the model
y_preds = model.predict(X_test)
np.mean(y_preds == y_test)
# ?????
```




    0.0




```python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_preds)
```




    2.6503039215686295



## Evaluation

- [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)

There are 3 different APIs for evaluating the quality of a model’s predictions:

- Estimator `score()` method
- The `scoring` parameter;  `cross-validation`
- Problem Specific Metric functions

### `model.score(X_test, y_test)`


```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("heart-disease.csv")
np.random.seed(0)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# instantiate the model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)


```

    (242, 13) (61, 13) (242,) (61,)





    RandomForestClassifier()




```python
# check the score on the test set
clf.score(X_test, y_test)
```




    0.8852459016393442




```python
df = pd.read_csv("Boston.csv",index_col=0)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
np.random.seed(0)
X = df.drop('medv', axis=1)
y = df['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

regr = RandomForestRegressor(n_estimators=100)
regr.fit(X_train, y_train)

```

    (404, 13) (102, 13) (404,) (102,)





    RandomForestRegressor()




```python
regr.score(X_test, y_test)
```




    0.7734908201180223



### cross validation with `cross_val_score()`

<div align="center">
<img src="img/cv.jpg" alt="cv.jpg" width="700px">
</div>


```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("heart-disease.csv")
np.random.seed(0)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# instantiate the model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
```

    (242, 13) (61, 13) (242,) (61,)





    0.8852459016393442




```python
from sklearn.model_selection import cross_val_score
```


```python
cross_val_score(clf, X, y, cv=10)
```




    array([0.87096774, 0.80645161, 0.83870968, 0.9       , 0.86666667,
           0.76666667, 0.73333333, 0.9       , 0.73333333, 0.8       ])




```python
np.random.seed(0)

# single training and test split score
clf_single_score = clf.score(X_test, y_test)

# take the mean of 10-fold cross validation score
clf_cross_val_score = np.mean(cross_val_score(clf, X, y, cv=10))

# compare the two scores
print(clf_single_score, clf_cross_val_score)
```

    0.8852459016393442 0.8347311827956989


### Problem Specific Metrics for Classification

 - Accuracy
 - Area Under the ROC Curve
 - Confusion Matrix
 - Classification Report

#### Accuracy


```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("heart-disease.csv")
np.random.seed(0)
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
clf = RandomForestClassifier(n_estimators=100)

```

    (242, 13) (61, 13) (242,) (61,)



```python
cross_val_score = cross_val_score(clf, X, y, cv=10)
```


```python
np.mean(cross_val_score)
```




    0.8347311827956989




```python
print(f"Heart disease Cross-Validated Accuracy: {np.mean(cross_val_score) * 100:.2f}%")
```

    Heart disease Cross-Validated Accuracy: 83.47%


#### Area Under the ROC Curve

**Area Under the receiver operating characteristic curve (ROC/AUC)**

- Area under the cure(AUC)
- ROC curve

ROC curve are a comparison of the true positive rate (TPR) versus a models false positive rate (FPR).

- `True positive`  = model predicts `1` and truth is `1`
- `False positive` = model predicts `1` and truth is `0`
- `True negative`  = model predicts `0` and truth is `0`
- `False negative` = model predicts `0` and truth is `1`


```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("heart-disease.csv")
np.random.seed(0)
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
clf = RandomForestClassifier(n_estimators=100)

```

    (242, 13) (61, 13) (242,) (61,)



```python
from sklearn.metrics import roc_curve

# fit
clf.fit(X_train, y_train)

# make predictions with probabilities
y_probs = clf.predict_proba(X_test)
y_probs[:5]
```




    array([[0.86, 0.14],
           [0.46, 0.54],
           [0.4 , 0.6 ],
           [1.  , 0.  ],
           [0.84, 0.16]])




```python
y_probs_positive  = y_probs[:, 1]
y_probs_positive[:5]
```




    array([0.14, 0.54, 0.6 , 0.  , 0.16])




```python
# calculate fpr,tp,thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_probs_positive)
fpr
```




    array([0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.03703704, 0.03703704, 0.03703704, 0.14814815, 0.18518519,
           0.18518519, 0.18518519, 0.18518519, 0.22222222, 0.22222222,
           0.33333333, 0.37037037, 0.55555556, 0.55555556, 0.7037037 ,
           0.81481481, 0.88888889, 1.        ])




```python
def plot_roc_curve(fpr,tpr):
	"""
	Plots a ROC curve given the false positive rate and true positive rate.
	"""
	plt.plot(fpr, tpr, color='orange', label='ROC')
	plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='Guessing')
	# customise the plot
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic (ROC) Curve')
	plt.legend()
	plt.show()

plot_roc_curve(fpr,tpr)
```



![jpeg](README_files/README_69_0.jpg)




```python
# how a perfect ROC curve looks like?
ftr , tpr, thresholds = roc_curve(y_test, y_test)
plot_roc_curve(ftr,tpr)
```



![jpeg](README_files/README_70_0.jpg)




```python
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, y_probs_positive)

```




    0.9281045751633987





## Putting All Together: PipeLine + GridSearchCV

- [towardsdatascience.com/how-to-use-sklearn-pipelines-for-ridiculously-neat-code](https://towardsdatascience.com/how-to-use-sklearn-pipelines-for-ridiculously-neat-code-a61ab66ca90d)
- [towardsdatascience.com/are-you-using-pipeline-in-scikit-learn](https://towardsdatascience.com/are-you-using-pipeline-in-scikit-learn-ac4cd85cb27f)
- [https://medium.com/vickdata/a-simple-guide-to-scikit-learn-pipelines](https://medium.com/vickdata/a-simple-guide-to-scikit-learn-pipelines-4ac0d974bdcf)

### v1: Pipelining up to Single Estimator



```python
# getting data ready
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# modelling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# setup random seed for reproducibility
import numpy as np
np.random.seed(42)

```

#### 1. EDA


```python
df = pd.read_csv("house_prices.csv")
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>




```python
X = df.drop('SalePrice', axis=1)
y = df.SalePrice

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25,
                                                      random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

```

    (1095, 80) (365, 80) (1095,) (365,)



```python
X_train.describe().T.iloc[:10]

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Id</th>
      <td>1095.0</td>
      <td>733.713242</td>
      <td>421.940022</td>
      <td>1.0</td>
      <td>366.50</td>
      <td>747.0</td>
      <td>1099.5</td>
      <td>1460.0</td>
    </tr>
    <tr>
      <th>MSSubClass</th>
      <td>1095.0</td>
      <td>56.602740</td>
      <td>42.201335</td>
      <td>20.0</td>
      <td>20.00</td>
      <td>50.0</td>
      <td>70.0</td>
      <td>190.0</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>896.0</td>
      <td>69.764509</td>
      <td>23.116448</td>
      <td>21.0</td>
      <td>58.75</td>
      <td>69.5</td>
      <td>80.0</td>
      <td>313.0</td>
    </tr>
    <tr>
      <th>LotArea</th>
      <td>1095.0</td>
      <td>10554.273973</td>
      <td>10059.063819</td>
      <td>1300.0</td>
      <td>7734.00</td>
      <td>9531.0</td>
      <td>11592.0</td>
      <td>215245.0</td>
    </tr>
    <tr>
      <th>OverallQual</th>
      <td>1095.0</td>
      <td>6.071233</td>
      <td>1.363015</td>
      <td>1.0</td>
      <td>5.00</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>OverallCond</th>
      <td>1095.0</td>
      <td>5.568037</td>
      <td>1.115243</td>
      <td>1.0</td>
      <td>5.00</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>YearBuilt</th>
      <td>1095.0</td>
      <td>1971.006393</td>
      <td>30.205435</td>
      <td>1872.0</td>
      <td>1954.00</td>
      <td>1972.0</td>
      <td>2000.0</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>YearRemodAdd</th>
      <td>1095.0</td>
      <td>1984.691324</td>
      <td>20.577087</td>
      <td>1950.0</td>
      <td>1966.50</td>
      <td>1993.0</td>
      <td>2003.0</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>1090.0</td>
      <td>104.244037</td>
      <td>183.990710</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>170.0</td>
      <td>1600.0</td>
    </tr>
    <tr>
      <th>BsmtFinSF1</th>
      <td>1095.0</td>
      <td>441.032877</td>
      <td>434.599451</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>381.0</td>
      <td>716.0</td>
      <td>2260.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train.describe(include=object).T.iloc[:10]  # All object cols

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MSZoning</th>
      <td>1095</td>
      <td>5</td>
      <td>RL</td>
      <td>873</td>
    </tr>
    <tr>
      <th>Street</th>
      <td>1095</td>
      <td>2</td>
      <td>Pave</td>
      <td>1090</td>
    </tr>
    <tr>
      <th>Alley</th>
      <td>65</td>
      <td>2</td>
      <td>Grvl</td>
      <td>33</td>
    </tr>
    <tr>
      <th>LotShape</th>
      <td>1095</td>
      <td>4</td>
      <td>Reg</td>
      <td>688</td>
    </tr>
    <tr>
      <th>LandContour</th>
      <td>1095</td>
      <td>4</td>
      <td>Lvl</td>
      <td>991</td>
    </tr>
    <tr>
      <th>Utilities</th>
      <td>1095</td>
      <td>2</td>
      <td>AllPub</td>
      <td>1094</td>
    </tr>
    <tr>
      <th>LotConfig</th>
      <td>1095</td>
      <td>5</td>
      <td>Inside</td>
      <td>795</td>
    </tr>
    <tr>
      <th>LandSlope</th>
      <td>1095</td>
      <td>3</td>
      <td>Gtl</td>
      <td>1030</td>
    </tr>
    <tr>
      <th>Neighborhood</th>
      <td>1095</td>
      <td>25</td>
      <td>NAmes</td>
      <td>163</td>
    </tr>
    <tr>
      <th>Condition1</th>
      <td>1095</td>
      <td>9</td>
      <td>Norm</td>
      <td>959</td>
    </tr>
  </tbody>
</table>
</div>




```python
above_0_missing = X_train.isnull().sum() > 0
missing = X_train.isnull().sum()[above_0_missing]
missing.shape,missing
```




    ((19,),
     LotFrontage      199
     Alley           1030
     MasVnrType         5
     MasVnrArea         5
     BsmtQual          25
     BsmtCond          25
     BsmtExposure      25
     BsmtFinType1      25
     BsmtFinType2      26
     Electrical         1
     FireplaceQu      521
     GarageType        56
     GarageYrBlt       56
     GarageFinish      56
     GarageQual        56
     GarageCond        56
     PoolQC          1091
     Fence            896
     MiscFeature     1048
     dtype: int64)



19 features have NaNs

#### 2. Define numerical and categorical features.


```python
numerical_features = X_train.select_dtypes(include='number').columns.tolist()
print(f'There are {len(numerical_features)} numerical features:', '\n')
print(numerical_features)
```

    There are 37 numerical features:

    ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']



```python
categorical_features = X_train.select_dtypes(exclude='number').columns.tolist()
print(f'There are {len(categorical_features)} categorical features:', '\n')
print(categorical_features)
```

    There are 43 categorical features:

    ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']


#### 3. PipeLine 1: For categorical features: fill missing values then label encode

For categoricals, we will use `SimpleImputer` to fill the missing values with the `mode`(value that appears most often) of each column. And then convert the categorical columns to numerical columns using `OneHotEncoder`.


```python
categorical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

```

> Set `handle_unknown` to `ignore` to skip previously unseen labels. Otherwise, `OneHotEncoder` throws an error if there are labels in test set that are not in train set.

#### 4. PipeLine 2: For numerical features: fill missing values then feature scale

For numeric columns, we first fill the missing values with `SimpleImputer` using the `mean` and feature scale using `MinMaxScaler`.


```python
numeric_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', MinMaxScaler())
])

```

#### 5. Combine numerical and categorical transformer using `ColumnTransformer`.

By default, all `Pipeline` objects have `fit` and `transform` methods which can be used to transform the input array like this:


```python
numeric_pipeline.fit_transform(X_train.select_dtypes(include='number'))
```




    array([[0.88553804, 0.29411765, 0.13356164, ..., 0.        , 1.        ,
            0.75      ],
           [0.69773818, 0.35294118, 0.16700174, ..., 0.        , 0.36363636,
            0.25      ],
           [0.83139136, 0.35294118, 0.16700174, ..., 0.        , 0.36363636,
            0.        ],
           ...,
           [0.83344757, 0.41176471, 0.1609589 , ..., 0.        , 0.27272727,
            1.        ],
           [0.38313914, 0.58823529, 0.16700174, ..., 0.        , 0.81818182,
            0.        ],
           [0.46881426, 0.23529412, 0.12671233, ..., 0.        , 0.45454545,
            1.        ]])



But, using the pipelines in this way means we have to call each pipeline separately on selected columns which is not what we want. What we want is to have a single preprocessor that is able to perform both numeric and categorical transformations in a single line of code like this: `full_processor.fit_transform(X_train)`

`ColumnTransformer` helps to define different transformers for different types of inputs and combine them into a single feature space after transformation. Here we are applying numerical transformer and categorical transformer created above for our numerical and categorical features.


```python
data_transformers = ColumnTransformer(transformers=[
    ('number', numeric_pipeline, numerical_features),
    ('category', categorical_pipeline, categorical_features)
])

```

> Remember that `numerical_features` and `categorical_features` contain the respective names of columns from `X_train`.

Similar to `Pipeline` class, `ColumnTransformer` takes a tuple of transformers. Each tuple `("step_name",transformer,columns)`should contain an arbitrary step name, the transformer itself and the list of column names that the transformer should be applied to . Here, we are creating a column transformer with 2 steps using both of our numeric and categorical preprocessing pipelines

#### 6. (optional) Apply PCA to reduce dimensions.

Principle Component Analysis aka PCA is a linear dimensionality reduction algorithm that is used to reduce the number of features in the dataset by keeping the maximum variance.


```python
#  Creating preprocessor pipeline which will first transform the data
# and then apply PCA.
preprocessor = Pipeline(steps=[('data_transformers', data_transformers),
                             ('reduce_dim',PCA())])
```

#### 7. Final Pipeline With an Estimator


```python
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', RandomForestRegressor())])
```


```python
model.fit(X_train, y_train)
model.score(X_test, y_test)
```




    0.6497809474139162




```python

```

#### Visualizing Pipeline


```python
# imports
from sklearn import set_config                      # to change the display
from sklearn.utils import estimator_html_repr       # to save the diagram into HTML format

# set config to diagram for visualizing the pipelines/composite estimators
set_config(display='diagram')

model

```








## Resource


```python

```
