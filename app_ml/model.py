import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os


def create_pipeline():
    # Load the dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'student_version.csv')
    data = pd.read_csv(data_path)
    print(data.shape)

    # Split the data into features and target
    X = data.drop(columns=['HeartDisease'])
    y = data['HeartDisease']
    print(X.shape, y.shape)

    # Split the data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


    # Define numerical and categorical columns
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    model = LogisticRegression(random_state=42)

    # Bundle preprocessing and modeling code in a pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)
                              ])

    # Fit the model
    pipeline.fit(X_train, y_train)
    print('Model trained')




    return pipeline



# def create_pipeline():
#     # Load the dataset
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     data_path = os.path.join(base_dir, '..', 'student_version.csv')
#     data = pd.read_csv(data_path)
#     print(data.shape)

#     # Split the data into features and target
#     X = data.drop(columns=['HeartDisease'])
#     y = data['HeartDisease']
#     print(X.shape, y.shape)

#     # Split the data into training, validation, and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
#     X_train, X_val, y_train, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


#     # Define numerical and categorical columns
#     numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
#     categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

#     # Preprocessing for numerical data
#     numerical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='constant')),
#         ('scaler', StandardScaler())
#     ])

#     # Preprocessing for categorical data
#     categorical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('onehot', OneHotEncoder(handle_unknown='ignore'))
#     ])

#     # Bundle preprocessing for numerical and categorical data
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numerical_transformer, numerical_cols),
#             ('cat', categorical_transformer, categorical_cols)
#         ])

#     model = LogisticRegression(random_state=42)

#     # Bundle preprocessing and modeling code in a pipeline
#     pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                                ('model', model)
#                               ])

#     # Fit the model
#     pipeline.fit(X_train, y_train)
#     print('Model trained')




#     return pipeline
model = create_pipeline()



# def create_pipeline():
#     # Load the dataset
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     data_path = os.path.join(base_dir, '..', 'student_version.csv')
#     data = pd.read_csv(data_path)
#     print(data.shape)

#     # Split the data into features and target
#     X = data.drop(columns=['HeartDisease'])
#     y = data['HeartDisease']
#     print(X.shape, y.shape)

#     # Split the data into training, validation, and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
#     X_train, X_val, y_train, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


#     # Define numerical and categorical columns
#     numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
#     categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

#     # Preprocessing for numerical data
#     numerical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='constant')),
#         ('scaler', StandardScaler())
#     ])

#     # Preprocessing for categorical data
#     categorical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('onehot', OneHotEncoder(handle_unknown='ignore'))
#     ])

#     # Bundle preprocessing for numerical and categorical data
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numerical_transformer, numerical_cols),
#             ('cat', categorical_transformer, categorical_cols)
#         ])

#     model = RandomForestClassifier(random_state=42)
#     # Bundle preprocessing and modeling code in a pipeline
#     pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                                ('model', model)
#                               ])

#     # Fit the model
#     pipeline.fit(X_train, y_train)
#     print('Model trained')




#     return pipeline