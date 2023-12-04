import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error


# Project 2-2
# 특정 년도의 타자의 해당 시즌 연봉을 예측하기 위한 다양한 ML 모델 학습
# 오름차순으로 year 칼럼으로 전체 데이터 정렬하는 함수
def sort_dataset(dataset_df):
    return dataset_df.sort_values(by='year', ascending=True)


# train/test datasets으로 전체 데이터 split
def split_dataset(dataset_df):
    train_indices = int(len(dataset_df) * 0.8)
    X = dataset_df.drop('salary', axis=1)
    y = dataset_df['salary'] * 0.001  # label 값 rescale
    X_train, X_test = X[:train_indices], X[train_indices:]
    Y_train, Y_test = y[:train_indices], y[train_indices:]
    return X_train, X_test, Y_train, Y_test


# numerical 값만 추출하기
def extract_numerical_cols(dataset_df):
    numerical_columns = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI',
                         'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']
    return dataset_df[numerical_columns]


# decision tree, random forest, svm에 대해서 함수를 훈련시키고 결과를 예측하는 함수
def train_predict_decision_tree(X_train, Y_train, X_test):
    decision_tree_regressor = DecisionTreeRegressor(random_state=41)
    decision_tree_regressor.fit(X_train, Y_train)
    return decision_tree_regressor.predict(X_test)


# Random Forest Regressor 모델 훈련시키고 결과 예측하기 위한 함수
def train_predict_random_forest(X_train, Y_train, X_test):
    random_forest_regressor = RandomForestRegressor(random_state=42)
    random_forest_regressor.fit(X_train, Y_train)
    return random_forest_regressor.predict(X_test)


# SVM 모델 훈련시키고 결과 예측하기 위한 함수
def train_predict_svm(X_train, Y_train, X_test):
    svm_pipeline = make_pipeline(StandardScaler(), SVR())
    svm_pipeline.fit(X_train, Y_train)
    return svm_pipeline.predict(X_test)


# RMSE 계산하고 리턴하는 함수
def calculate_RMSE(labels, predictions):
    return np.sqrt(mean_squared_error(labels, predictions))


# Main 함수
if __name__ == '__main__':
    # Load the dataset (path may need to be changed based on the file location)
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv', encoding='ANSI')

    sorted_df = sort_dataset(data_df)
    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)

    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)

    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)

    print("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
    print("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
    print("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))



