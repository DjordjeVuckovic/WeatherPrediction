from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import datasets
import statsmodels.api as sm
from scipy.stats import norm
import numpy as np

dataset = pd.read_csv('dataset/final_dataset.csv')


def model_accuracy(y_predv, y_test, n, d):
    mse = mean_squared_error(y_test, y_predv)
    mae = mean_absolute_error(y_test, y_predv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predv)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - d - 1)

    print('Mean squared error: ', mse)
    print('Mean absolute error: ', mae)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)
    print('R2 adjusted score: ', r2_adj)

    res = pd.concat([pd.DataFrame(y_test.values), pd.DataFrame(y_predv)], axis=1)
    res.columns = ['y', 'y_pred']
    print(res.head(30))


# values between 0 and 1 scale -MinMax
def scale_data(x, scale_mode):
    scaler = StandardScaler()
    if scale_mode == "standard":
        scaler = StandardScaler()
    if scale_mode == "min_max":
        scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(x)
    scaled_df = pd.DataFrame(scaled_data)
    return scaled_df


def linear_regression(X, y):
    # scale
    X = scale_data(X, "min-max")
    print(X.head())
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=42)
    # linear regression
    # Create a LinearRegression model
    model = LinearRegression(fit_intercept=True)
    # Fit the model to the training data
    model.fit(x_train, y_train)
    # Evaluate the model on the validation data
    val_score = model.score(x_val, y_val)
    print(f'Validation score: {val_score:.2f}')
    test_score = model.score(x_test, y_test)
    print(f'Test score: {test_score:.2f}')
    print("Coefs: ", model.coef_)
    y_predict = model.predict(x_test)
    model_accuracy(y_predict, y_test, x_test.shape[0], x_test.shape[1])


def ridge_regression(X, y, alpha_val=10):
    X = scale_data(X, "min-max")
    print(X.head())
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=42)
    model = Ridge(alpha=alpha_val)
    # Fit the model to the training data
    model.fit(x_train, y_train)
    # Evaluate the model on the validation data
    val_score = model.score(x_val, y_val)
    print(f'Validation score: {val_score:.2f}')
    test_score = model.score(x_test, y_test)
    print(f'Test score: {test_score:.2f}')
    y_predict = model.predict(x_test)
    print("Coefs: ", model.coef_)
    model_accuracy(y_predict, y_test, x_test.shape[0], x_test.shape[1])


def lasso_regression(X, y, alpha_val=0.01):
    X = scale_data(X, "min-max")
    print(X.head())
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=42)
    model = Lasso(alpha=alpha_val)
    # Fit the model to the training data
    model.fit(x_train, y_train)
    # Evaluate the model on the validation data
    val_score = model.score(x_val, y_val)
    print(f'Validation score: {val_score:.2f}')
    test_score = model.score(x_test, y_test)
    print(f'Test score: {test_score:.2f}')
    y_predict = model.predict(x_test)
    print("Coefs: ", model.coef_)
    model_accuracy(y_predict, y_test, x_test.shape[0], x_test.shape[1])


X = dataset.drop(['PM_US Post'], axis=1)
y = dataset['PM_US Post']
linear_regression(X, y)
# only correlated features
print("Correlated features")
X_corr = dataset[['season', 'PRES', 'cv', 'HUMI']]
y_corr = dataset['PM_US Post']
linear_regression(X_corr, y_corr)
print("Ridge reg")
ridge_regression(X, y)
print("Lasso reg")
lasso_regression(X, y)
# foward selection
print("Forward  selection")


# Define a function to evaluate the model performance
def evaluate_model(X_train, y_train, X_test, y_test, features):
    model = LinearRegression()
    model.fit(X_train[features], y_train)
    y_pred = model.predict(X_test[features])
    return mean_squared_error(y_test, y_pred)


# Initialize an empty list to store the selected features
selected_features = []
# Iterate over the full set of features
for feature in X.columns:
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=10)
    # Train the model using the current set of selected features, plus the feature being considered
    features = selected_features + [feature]
    # Evaluate the model performance
    mse = evaluate_model(x_train, y_train, x_test, y_test, features)
    # If the performance is improved, add the feature to the list of selected features
    best_mse = float("inf")
    if mse < best_mse:
        selected_features.append(feature)
        best_mse = mse

# Return the final list of selected features
print(selected_features)
print("Linear regression with forward selection")
X_forward = dataset[['year', 'month', 'day', 'hour', 'season', 'DEWP', 'HUMI', 'PRES', 'TEMP', 'Iws'
    , 'precipitation', 'Iprec', 'NE', 'NW', 'SE', 'SW', 'cv']]
y_forward = dataset['PM_US Post']
linear_regression(X_forward, y_forward)
